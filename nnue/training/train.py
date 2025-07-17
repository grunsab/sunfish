#!/usr/bin/env python3
"""
Training script for Sunfish NNUE model.
Trains on CCRL dataset with supervised learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from pathlib import Path
import argparse
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from nnue_model import SunfishNNUE

class ChessPositionDataset(Dataset):
    """Dataset for chess positions with evaluations."""
    
    def __init__(self, data_file, max_samples=None):
        """Load chess positions from JSON file."""
        
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        if max_samples and len(self.data) > max_samples:
            # Randomly sample positions for faster training
            np.random.shuffle(self.data)
            self.data = self.data[:max_samples]
        
        print(f"Loaded {len(self.data)} positions from {data_file}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Board string and target evaluation
        board_str = item['board']
        target_value = float(item['value'])
        
        return {
            'board': board_str,
            'target': torch.tensor(target_value, dtype=torch.float32)
        }

class ChessTrainer:
    """Training class for NNUE model."""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        
    def compute_features_batch(self, board_strings):
        """Compute features for a batch of board strings."""
        batch_wf = []
        batch_bf = []
        
        for board_str in board_strings:
            wf, bf = self.model.features_from_board(board_str)
            batch_wf.append(wf)
            batch_bf.append(bf)
        
        batch_wf = torch.stack(batch_wf).to(self.device)
        batch_bf = torch.stack(batch_bf).to(self.device)
        
        return batch_wf, batch_bf
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """Train for one epoch."""
        
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # Compute features
            wf, bf = self.compute_features_batch(batch['board'])
            targets = batch['target'].to(self.device)
            
            # Forward pass
            predictions = self.model(wf, bf).squeeze()
            
            # Compute loss
            loss = criterion(predictions, targets)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader, criterion):
        """Validate the model."""
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Compute features
                wf, bf = self.compute_features_batch(batch['board'])
                targets = batch['target'].to(self.device)
                
                # Forward pass
                predictions = self.model(wf, bf).squeeze()
                
                # Compute loss
                loss = criterion(predictions, targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def plot_losses(self, save_path=None):
        """Plot training and validation losses."""
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            print(f"Loss plot saved to {save_path}")
        else:
            plt.show()

def train_model(train_data_file, val_data_file, 
                model_size="small", 
                epochs=100, 
                batch_size=256,
                learning_rate=0.001,
                device='cpu',
                save_dir="models",
                max_train_samples=None,
                max_val_samples=None):
    """Main training function."""
    
    # Create save directory
    Path(save_dir).mkdir(exist_ok=True)
    
    # Model size configurations
    size_configs = {
        "tiny": {"L0": 6, "L1": 6, "L2": 6},
        "small": {"L0": 8, "L1": 8, "L2": 8}, 
        "medium": {"L0": 10, "L1": 10, "L2": 10},
        "large": {"L0": 16, "L1": 16, "L2": 16}
    }
    
    config = size_configs.get(model_size, size_configs["small"])
    print(f"Using {model_size} model: {config}")
    
    # Create model
    model = SunfishNNUE(**config)
    
    # Create datasets
    train_dataset = ChessPositionDataset(train_data_file, max_train_samples)
    val_dataset = ChessPositionDataset(val_data_file, max_val_samples)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Create trainer
    trainer = ChessTrainer(model, device)
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    print(f"Training on {len(train_dataset)} positions")
    print(f"Validating on {len(val_dataset)} positions")
    print(f"Device: {device}")
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    
    # Training loop
    for epoch in range(epochs):
        start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)
        
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer, criterion)
        
        # Validate
        val_loss = trainer.validate(val_loader, criterion)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        epoch_time = time.time() - start_time
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Time: {epoch_time:.2f}s")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            model_path = Path(save_dir) / f"best_model_{model_size}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'epoch': epoch,
                'val_loss': val_loss
            }, model_path)
            
            # Export to sunfish format
            sunfish_path = Path(save_dir) / f"best_model_{model_size}.pickle"
            model.export_to_sunfish_format(sunfish_path)
            
            print(f"New best model saved! Val loss: {val_loss:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = Path(save_dir) / f"checkpoint_{model_size}_epoch_{epoch+1}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses
            }, checkpoint_path)
    
    # Plot training progress
    plot_path = Path(save_dir) / f"training_progress_{model_size}.png"
    trainer.plot_losses(plot_path)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved in: {save_dir}")

def main():
    parser = argparse.ArgumentParser(description="Train Sunfish NNUE model")
    parser.add_argument("--train-data", required=True, help="Training data JSON file")
    parser.add_argument("--val-data", required=True, help="Validation data JSON file")
    parser.add_argument("--model-size", default="small", choices=["tiny", "small", "medium", "large"],
                        help="Model size configuration")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device to train on (cpu/cuda)")
    parser.add_argument("--save-dir", default="models", help="Directory to save models")
    parser.add_argument("--max-train-samples", type=int, help="Max training samples (for testing)")
    parser.add_argument("--max-val-samples", type=int, help="Max validation samples (for testing)")
    
    args = parser.parse_args()
    
    # Check if data files exist
    if not Path(args.train_data).exists():
        print(f"Training data file not found: {args.train_data}")
        print("Run preprocess_data.py first to create training data.")
        return
    
    if not Path(args.val_data).exists():
        print(f"Validation data file not found: {args.val_data}")
        print("Run preprocess_data.py first to create training data.")
        return
    
    train_model(
        train_data_file=args.train_data,
        val_data_file=args.val_data,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        save_dir=args.save_dir,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples
    )

if __name__ == "__main__":
    main()