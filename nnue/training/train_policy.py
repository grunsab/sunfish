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
from multiprocessing import Pool, cpu_count
from functools import partial

# Unbuffer output
sys.stdout.flush()
print("Starting train_policy.py...", flush=True)

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    import chess
    import chess.pgn
    print("All imports successful", flush=True)
except ImportError as e:
    print(f"Import error: {e}", flush=True)
    sys.exit(1)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
print(f"Added to path: {Path(__file__).parent.parent.parent}", flush=True)

try:
    from policy_network import SunfishPolicyNetwork, PolicyTrainer, create_policy_network
    from sunfish_nnue_base import Position, Move, initial
    print("Local imports successful", flush=True)
except ImportError as e:
    print(f"Local import error: {e}", flush=True)
    sys.exit(1)

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
    
    pgn_file_name = os.path.basename(pgn_file)
    training_examples = []
    
    try:
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
                
                if len(moves) < 20:  # Skip very short games
                    continue
                
                # Convert chess.Board to sunfish format and extract training examples
                try:
                    examples = extract_training_examples_from_game(board, moves)
                    training_examples.extend(examples)
                except Exception as e:
                    continue
                
                if game_count % 100 == 0:
                    print(f"    [{pgn_file_name}] Processed {game_count} games, extracted {len(training_examples)} examples")
        
        print(f"  [{pgn_file_name}] Finished: {game_count} games, {len(training_examples)} examples")
        
    except Exception as e:
        print(f"  Error reading PGN file {pgn_file}: {e}")
    
    return training_examples

def process_pgn_file_worker(args):
    """Worker function for multiprocessing"""
    pgn_file, max_games_per_file = args
    return parse_pgn_games(str(pgn_file), max_games_per_file)

def sunfish_pos_to_square(pos):
    """Convert sunfish position (21-98) to 0-63 square index"""
    if pos < 21 or pos > 98:
        return None
    
    # Calculate rank and file from sunfish position
    rank = (pos - 21) // 10
    file = (pos - 21) % 10
    
    # Check if it's a valid board position
    if rank < 0 or rank > 7 or file < 0 or file > 7:
        return None
    
    # Convert to 0-63 index (a1=0, h8=63)
    return (7 - rank) * 8 + file

def move_to_index(move):
    """Convert Sunfish Move object to policy network index
    This must match the indexing scheme in policy_network.py exactly
    """
    from_sq = sunfish_pos_to_square(move.i)
    to_sq = sunfish_pos_to_square(move.j)
    
    if from_sq is None or to_sq is None:
        return None
    
    # Build the same mapping as policy_network.py
    index = 0
    
    # Regular moves (from square to square)
    for f in range(64):
        for t in range(64):
            if f != t:
                if f == from_sq and t == to_sq and not move.prom:
                    return index
                index += 1
    
    # Promotion moves
    for f in range(64):
        for t in range(64):
            if f != t:
                for prom in ['N', 'B', 'R', 'Q']:
                    if f == from_sq and t == to_sq and move.prom == prom:
                        return index
                    index += 1
    
    return None

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
        # Convert board to sunfish format
        sunfish_board = chess_board_to_sunfish(board)
        
        # Convert move to sunfish format
        sunfish_move = chess_move_to_sunfish(move, board)
        
        if sunfish_move:
            # Calculate target index for the move
            target_index = move_to_index(sunfish_move)
            
            if target_index is not None:
                examples.append({
                    'board': sunfish_board,
                    'move': sunfish_move,
                    'target_index': target_index,
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

def create_training_data_from_ccrl(data_dir, output_file, max_games_per_file=100, use_train_data=True, num_workers=None):
    """Create training data from CCRL data structure"""
    print(f"\n=== Creating Training Data from CCRL ===")
    print(f"Data directory: {data_dir}")
    print(f"Max games per file: {max_games_per_file}")
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    print(f"Using {num_workers} worker processes")
    
    all_examples = []
    
    # Use train or test data
    subdir = "train" if use_train_data else "test"
    pgn_dir = Path(data_dir).resolve() / "cclr" / subdir
    
    print(f"Looking for PGN files in: {pgn_dir}")
    
    if not pgn_dir.exists():
        print(f"Error: Directory {pgn_dir} does not exist")
        return
    
    # Process all PGN files in the directory
    pgn_files = sorted(list(pgn_dir.glob("*.pgn")))  # Remove the [:5] limitation
    print(f"Found {len(pgn_files)} PGN files to process")
    
    # Prepare arguments for workers
    worker_args = [(pgn_file, max_games_per_file) for pgn_file in pgn_files]
    
    # Process files in parallel
    print(f"\nProcessing {len(pgn_files)} files in parallel...")
    start_time = time.time()
    
    with Pool(num_workers) as pool:
        # Use map with chunksize for better performance
        chunksize = max(1, len(pgn_files) // (num_workers * 4))
        results = pool.map(process_pgn_file_worker, worker_args, chunksize=chunksize)
    
    # Combine results from all workers
    for i, examples in enumerate(results):
        all_examples.extend(examples)
        if (i + 1) % 10 == 0:
            print(f"  Combined results from {i + 1}/{len(pgn_files)} files, total examples: {len(all_examples)}")
    
    elapsed_time = time.time() - start_time
    print(f"\nProcessing completed in {elapsed_time:.2f} seconds")
    print(f"Total examples extracted: {len(all_examples)}")
    
    if len(all_examples) == 0:
        print("Warning: No examples were extracted. Check your PGN files.")
        return
    
    # Target indices are now calculated during game processing
    print("Target indices calculated during game processing")
    
    # Save to JSON
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(all_examples, f, indent=2, default=str)
    
    print(f"Successfully saved training data to {output_file}")

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
    parser.add_argument('--num-workers', type=int, default=None, help='Number of worker processes (default: number of CPU cores)')
    
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
    print(f"Arguments: {args}")
    if args.create_data:
        use_train_data = not args.use_test_data
        create_training_data_from_ccrl(args.data_dir, args.output_data, args.max_games_per_file, use_train_data, args.num_workers)
    else:
        train_policy_network(args)

if __name__ == "__main__":
    print("Executing main function...", flush=True)
    main()