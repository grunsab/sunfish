#!/usr/bin/env python3
"""
Training script for the policy network that learns from played moves.
This implements the core concept: policy = next_move_played
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import chess
import chess.pgn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from policy_network import SunfishPolicyNetwork, PolicyTrainer, create_policy_network
from sunfish import Position, Move, initial

class PolicyDataset(Dataset):
    """Dataset for policy network training"""
    
    def __init__(self, data_file, max_samples=None):
        self.examples = []
        self.load_data(data_file, max_samples)
    
    def load_data(self, data_file, max_samples):
        """Load training data from JSON file"""
        print(f"Loading data from {data_file}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        self.examples = data
        print(f"Loaded {len(self.examples)} training examples")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Convert to tensors
        board_tensor = torch.zeros(16, 8, 8)  # Will be populated by network
        target_index = example['target_index']
        
        return {
            'board_string': example['board'],
            'target_index': target_index,
            'move': example['move']
        }

def parse_pgn_games(pgn_file, max_games=None):
    """Parse PGN games and extract training positions"""
    
    training_examples = []
    
    with open(pgn_file, 'r') as f:
        game_count = 0
        
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            game_count += 1
            if max_games and game_count > max_games:
                break
            
            # Extract positions and moves from the game
            board = game.board()
            moves = list(game.mainline_moves())
            
            # Convert chess.Board to sunfish format and extract training examples
            try:
                examples = extract_training_examples_from_game(board, moves)
                training_examples.extend(examples)
            except Exception as e:
                print(f"Error processing game {game_count}: {e}")
                continue
            
            if game_count % 1000 == 0:
                print(f"Processed {game_count} games, extracted {len(training_examples)} examples")
    
    return training_examples

def chess_board_to_sunfish(board):
    """Convert chess.Board to sunfish board string"""
    # This is a simplified conversion - in practice you'd need more sophisticated mapping
    sunfish_board = list(initial)
    
    # Map chess positions to sunfish positions
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            # Convert to sunfish coordinates (this is simplified)
            sunfish_pos = 21 + (7 - rank) * 10 + file
            
            piece_symbol = piece.symbol()
            sunfish_board[sunfish_pos] = piece_symbol
    
    return ''.join(sunfish_board)

def extract_training_examples_from_game(board, moves):
    """Extract training examples from a chess game"""
    examples = []
    
    for i, move in enumerate(moves):
        # Skip opening and endgame moves
        if i < 10 or i > len(moves) - 10:
            continue
        
        # Convert board to sunfish format
        sunfish_board = chess_board_to_sunfish(board)
        
        # Convert move to sunfish format
        sunfish_move = chess_move_to_sunfish(move, board)
        
        if sunfish_move:
            examples.append({
                'board': sunfish_board,
                'move': sunfish_move,
                'game_phase': 'middle' if 10 <= i <= len(moves) - 10 else 'other'
            })
        
        # Make the move
        board.push(move)
    
    return examples

def chess_move_to_sunfish(chess_move, board):
    """Convert chess.Move to sunfish Move (simplified)"""
    # This is a placeholder - real implementation would need proper coordinate mapping
    from_square = chess_move.from_square
    to_square = chess_move.to_square
    
    # Convert chess squares to sunfish positions
    from_file = chess.square_file(from_square)
    from_rank = chess.square_rank(from_square)
    to_file = chess.square_file(to_square)
    to_rank = chess.square_rank(to_square)
    
    # Convert to sunfish coordinates
    sunfish_from = 21 + (7 - from_rank) * 10 + from_file
    sunfish_to = 21 + (7 - to_rank) * 10 + to_file
    
    promotion = ""
    if chess_move.promotion:
        promotion = chess.piece_name(chess_move.promotion).upper()[0]
    
    return Move(sunfish_from, sunfish_to, promotion)

def create_training_data_from_ccrl(data_dir, output_file, max_games_per_file=100, use_train_data=True):
    """Create training data from CCRL data structure"""
    all_examples = []
    
    # Use train or test data
    subdir = "train" if use_train_data else "test"
    pgn_dir = Path(data_dir) / "cclr" / subdir
    
    if not pgn_dir.exists():
        print(f"Error: Directory {pgn_dir} does not exist")
        return
    
    # Process all PGN files in the directory
    pgn_files = list(pgn_dir.glob("*.pgn"))
    print(f"Found {len(pgn_files)} PGN files in {pgn_dir}")
    
    for pgn_file in pgn_files:
        print(f"Processing {pgn_file}")
        examples = parse_pgn_games(str(pgn_file), max_games_per_file)
        all_examples.extend(examples)
        
        # Progress update
        if len(all_examples) % 10000 == 0:
            print(f"Processed {len(all_examples)} examples so far...")
    
    print(f"Total examples: {len(all_examples)}")
    
    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(all_examples, f, indent=2, default=str)
    
    print(f"Saved training data to {output_file}")

def train_policy_network(args):
    """Train the policy network"""
    
    # Create network
    device = torch.device('cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"Using device: {device}")
    
    network = create_policy_network(args.model_size)
    trainer = PolicyTrainer(network, learning_rate=args.learning_rate, device=device)
    
    # Load existing model if specified
    if args.resume:
        trainer.load_model(args.resume)
        print(f"Resumed from {args.resume}")
    
    # Load training data
    if args.train_data_json:
        dataset = PolicyDataset(args.train_data_json, args.max_train_samples)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    else:
        print("No training data specified")
        return
    
    print(f"Starting training for {args.epochs} epochs")
    
    # Training loop
    for epoch in range(args.epochs):
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            # Add training examples to trainer
            for i in range(len(batch['board_string'])):
                board_string = batch['board_string'][i]
                target_index = batch['target_index'][i].item()
                
                # Create dummy move object (simplified)
                move = Move(0, 0, "")  # This would need proper implementation
                
                trainer.add_training_example(board_string, move)
            
            # Train on accumulated examples
            loss = trainer.train_step()
            if loss is not None:
                total_loss += loss
                num_batches += 1
                progress_bar.set_postfix({'loss': f'{loss:.4f}'})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1} completed, average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = f"policy_checkpoint_epoch_{epoch+1}.pth"
            trainer.save_model(checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_model_path = f"policy_model_{args.model_size}_final.pth"
    trainer.save_model(final_model_path)
    print(f"Training completed! Final model saved to {final_model_path}")
    
    # Print training statistics
    stats = trainer.get_training_stats()
    print(f"Training statistics: {stats}")

def main():
    parser = argparse.ArgumentParser(description='Train policy network for Sunfish')
    
    # Data preparation
    parser.add_argument('--create-data', action='store_true', help='Create training data from CCRL PGN files')
    parser.add_argument('--data-dir', default='../../../data', help='Root directory containing CCRL data')
    parser.add_argument('--output-data', default='policy_training_data.json', help='Output file for training data')
    parser.add_argument('--max-games-per-file', type=int, default=10000000, help='Maximum games per PGN file')
    parser.add_argument('--use-train-data', action='store_true', default=True, help='Use training data (default: True)')
    parser.add_argument('--use-test-data', action='store_true', help='Use test data instead of training data')
    
    # Training
    parser.add_argument('--train-data-json', help='JSON file with training data')
    parser.add_argument('--model-size', choices=['tiny', 'small', 'medium', 'large'], default='small', help='Model size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device to use')
    parser.add_argument('--max-train-samples', type=int, help='Maximum training samples')
    parser.add_argument('--resume', help='Resume from checkpoint')
    parser.add_argument('--save-every', type=int, default=10, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    if args.create_data:
        use_train_data = not args.use_test_data
        create_training_data_from_ccrl(args.data_dir, args.output_data, args.max_games_per_file, use_train_data)
    else:
        train_policy_network(args)

if __name__ == "__main__":
    main()