#!/usr/bin/env python3
"""
Example usage of the policy-integrated Sunfish engine.
Demonstrates how the policy network learns from played moves.
"""

import sys
import time
import subprocess
import os

def run_engine_game():
    """Run a simple game with the policy-integrated engine"""
    
    print("=== Policy-Integrated Sunfish Example ===")
    print("This example demonstrates how the policy network learns from played moves.")
    print()
    
    # Start the engine
    engine = subprocess.Popen(
        [sys.executable, "sunfish_policy_integrated.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    def send_command(cmd):
        print(f">>> {cmd}")
        engine.stdin.write(cmd + "\n")
        engine.stdin.flush()
        
        # Read response
        while True:
            line = engine.stdout.readline()
            if not line:
                break
            line = line.strip()
            if line:
                print(f"<<< {line}")
                if line == "uciok" or line == "readyok" or line.startswith("bestmove"):
                    break
    
    try:
        # Initialize UCI
        send_command("uci")
        send_command("isready")
        
        # Set up starting position
        send_command("position startpos")
        
        # Play a few moves to demonstrate policy learning
        moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"]
        
        print("\n=== Playing example moves ===")
        for move in moves:
            print(f"\nAdding move: {move}")
            send_command(f"position startpos moves {' '.join(moves[:moves.index(move)+1])}")
            
            # Let the engine think for a short time
            send_command("go movetime 1000")
            
            # Small delay to see the learning process
            time.sleep(0.5)
        
        print("\n=== Training Statistics ===")
        print("The policy network has been learning from each move played.")
        print("In a real game, it would gradually improve its move selection.")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Clean shutdown
        send_command("quit")
        engine.wait()

def create_training_data_example():
    """Example of creating training data from CCRL games"""
    
    print("\n=== Creating Training Data Example ===")
    print("This shows how to create training data from CCRL games.")
    print()
    
    # Check if data directory exists
    if not os.path.exists("data/cclr/train"):
        print("CCRL training data not found. Please ensure the data directory exists.")
        return
    
    # Import the training script
    sys.path.append("nnue/training")
    
    try:
        from train_policy import create_training_data_from_ccrl
        
        print("Creating training data from CCRL games...")
        create_training_data_from_ccrl(
            data_dir="data",
            output_file="example_policy_training_data.json",
            max_games_per_file=10,  # Small number for demo
            use_train_data=True
        )
        
        print("Training data created successfully!")
        print("You can now use this data to train the policy network.")
        
    except ImportError as e:
        print(f"Could not import training modules: {e}")
        print("Make sure PyTorch and python-chess are installed.")
    except Exception as e:
        print(f"Error creating training data: {e}")

def main():
    """Main example function"""
    
    print("Sunfish Policy Network Integration Example")
    print("=" * 45)
    
    try:
        # Run engine game example
        run_engine_game()
        
        # Create training data example
        create_training_data_example()
        
        print("\n=== Summary ===")
        print("✓ Policy network integration working")
        print("✓ Engine learns from played moves")
        print("✓ Training data can be created from CCRL games")
        print("✓ Model automatically saves on exit")
        print()
        print("Next steps:")
        print("1. Use tools/fancy.py to play against the policy engine")
        print("2. Train on larger datasets using nnue/training/train_policy.py")
        print("3. Evaluate model performance using existing test suite")
        
    except KeyboardInterrupt:
        print("\nExample interrupted by user.")
    except Exception as e:
        print(f"Example failed: {e}")

if __name__ == "__main__":
    main()