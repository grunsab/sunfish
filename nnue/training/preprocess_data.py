#!/usr/bin/env python3
"""
Preprocess CCRL PGN data for NNUE training.
Converts PGN games to training positions with evaluations.
"""

import os
import sys
import chess
import chess.pgn
import chess.engine
import numpy as np
import pickle
from pathlib import Path
import json
from collections import defaultdict
import random

# Add parent directory to path to import sunfish
sys.path.append(str(Path(__file__).parent.parent.parent))

def board_to_sunfish_string(board):
    """Convert python-chess board to sunfish 120-character string format."""
    
    # Sunfish board layout (10x12 with padding)
    sunfish_board = [' '] * 120
    
    # Fill the center 8x8 area
    for square in chess.SQUARES:
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        
        # Convert to sunfish coordinates (rank 7 = index 2, rank 0 = index 9)
        sunfish_rank = 9 - rank_idx
        sunfish_file = file_idx + 1
        sunfish_idx = sunfish_rank * 10 + sunfish_file
        
        piece = board.piece_at(square)
        if piece:
            sunfish_board[sunfish_idx] = piece.symbol()
        else:
            sunfish_board[sunfish_idx] = '.'
    
    # Add newlines at the end of each rank
    for rank in range(10):
        sunfish_board[rank * 10 + 9] = '\n'
    
    return ''.join(sunfish_board)

def extract_training_positions(pgn_files, max_games=10000, max_positions_per_game=20):
    """Extract training positions from PGN files."""
    
    positions = []
    game_count = 0
    
    for pgn_file in pgn_files:
        print(f"Processing {pgn_file}...")
        
        with open(pgn_file) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                
                if game_count >= max_games:
                    break
                
                # Skip games without proper result
                result = game.headers.get("Result", "*")
                if result == "*":
                    continue
                
                # Convert result to score from white's perspective
                if result == "1-0":
                    game_result = 1.0
                elif result == "0-1":
                    game_result = -1.0
                else:  # Draw
                    game_result = 0.0
                
                # Extract positions from the game
                board = game.board()
                move_count = 0
                
                for move in game.mainline_moves():
                    if move_count >= max_positions_per_game:
                        break
                    
                    # Get position before move
                    sunfish_board = board_to_sunfish_string(board)
                    
                    # Use game result as position evaluation (simplified)
                    # In a more sophisticated approach, you'd use an engine to evaluate each position
                    position_value = game_result
                    if not board.turn:  # Black to move
                        position_value = -position_value
                    
                    positions.append({
                        'board': sunfish_board,
                        'value': position_value,
                        'turn': board.turn
                    })
                    
                    board.push(move)
                    move_count += 1
                
                game_count += 1
                if game_count % 1000 == 0:
                    print(f"Processed {game_count} games, {len(positions)} positions")
        
        if game_count >= max_games:
            break
    
    print(f"Extracted {len(positions)} positions from {game_count} games")
    return positions

def create_training_data(positions, output_dir="processed_data"):
    """Create training data files for NNUE."""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    print("Creating training data...")
    
    # Shuffle positions
    random.shuffle(positions)
    
    # Split into train/validation
    split_idx = int(0.9 * len(positions))
    train_positions = positions[:split_idx]
    val_positions = positions[split_idx:]
    
    # Save training data
    train_file = Path(output_dir) / "train_data.json"
    val_file = Path(output_dir) / "val_data.json"
    
    with open(train_file, 'w') as f:
        json.dump(train_positions, f)
    
    with open(val_file, 'w') as f:
        json.dump(val_positions, f)
    
    print(f"Saved {len(train_positions)} training positions to {train_file}")
    print(f"Saved {len(val_positions)} validation positions to {val_file}")
    
    # Create summary
    summary = {
        'total_positions': len(positions),
        'train_positions': len(train_positions),
        'val_positions': len(val_positions),
        'value_distribution': {
            'wins': sum(1 for p in positions if p['value'] > 0.5),
            'draws': sum(1 for p in positions if abs(p['value']) <= 0.5),
            'losses': sum(1 for p in positions if p['value'] < -0.5)
        }
    }
    
    summary_file = Path(output_dir) / "data_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Data summary saved to {summary_file}")
    print(f"Value distribution: {summary['value_distribution']}")
    
    return train_file, val_file

def main():
    """Main preprocessing pipeline."""
    
    data_dir = Path("data")
    if not data_dir.exists():
        print("Data directory not found. Run download_ccrl.py first.")
        return
    
    # Find PGN files
    pgn_files = list(data_dir.glob("*.pgn"))
    if not pgn_files:
        print("No PGN files found in data directory.")
        print("Please download CCRL PGN files first.")
        return
    
    print(f"Found {len(pgn_files)} PGN files:")
    for f in pgn_files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    
    # Extract positions
    print("\nExtracting training positions...")
    positions = extract_training_positions(pgn_files, max_games=50000)
    
    if not positions:
        print("No positions extracted. Check PGN files.")
        return
    
    # Create training data
    print("\nCreating training data files...")
    train_file, val_file = create_training_data(positions)
    
    print(f"\nPreprocessing complete!")
    print(f"Training data: {train_file}")
    print(f"Validation data: {val_file}")

if __name__ == "__main__":
    main()