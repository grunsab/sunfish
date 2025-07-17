#!/usr/bin/env python3
"""
Evaluate trained NNUE model against sunfish_nnue.py and existing models.
"""

import sys
import time
import subprocess
import tempfile
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

def test_model_compatibility(model_file):
    """Test if the model works with sunfish_nnue.py."""
    
    print(f"Testing model compatibility: {model_file}")
    
    if not Path(model_file).exists():
        print(f"Model file not found: {model_file}")
        return False
    
    # Test with sunfish_nnue.py
    sunfish_nnue_path = Path(__file__).parent.parent.parent / "sunfish_nnue.py"
    
    try:
        # Run a quick test position
        test_commands = [
            "uci",
            "isready", 
            "position startpos moves e2e4 e7e5",
            "go depth 3",
            "quit"
        ]
        
        process = subprocess.Popen(
            [sys.executable, str(sunfish_nnue_path), str(model_file)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        commands_str = "\n".join(test_commands) + "\n"
        stdout, stderr = process.communicate(input=commands_str, timeout=30)
        
        if process.returncode == 0:
            print("✓ Model is compatible with sunfish_nnue.py")
            
            # Check if we got a bestmove
            if "bestmove" in stdout:
                print("✓ Model successfully produces moves")
                lines = stdout.strip().split('\n')
                for line in lines:
                    if line.startswith("bestmove"):
                        print(f"  Sample move: {line}")
                        break
                return True
            else:
                print("✗ Model did not produce a move")
                return False
        else:
            print("✗ Model failed to run with sunfish_nnue.py")
            if stderr:
                print(f"Error: {stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Model test timed out")
        process.kill()
        return False
    except Exception as e:
        print(f"✗ Test failed with exception: {e}")
        return False

def compare_model_sizes(model_files):
    """Compare sizes of different models."""
    
    print("\nModel Size Comparison:")
    print("-" * 40)
    
    for model_file in model_files:
        if Path(model_file).exists():
            size_bytes = Path(model_file).stat().st_size
            size_kb = size_bytes / 1024
            print(f"{Path(model_file).name:25} {size_bytes:8} bytes ({size_kb:.1f} KB)")
        else:
            print(f"{Path(model_file).name:25} Not found")

def benchmark_models(model_files, depth=4, positions=5):
    """Benchmark model performance."""
    
    print(f"\nBenchmarking Models (depth={depth}, positions={positions}):")
    print("-" * 60)
    
    # Test positions (FEN format)
    test_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting position
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 4",  # Italian game
        "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 0 4",  # Queen's gambit
        "r1bq1rk1/ppp2ppp/2n2n2/2bpp3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 b - - 0 6",  # Middlegame
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"  # Endgame
    ][:positions]
    
    sunfish_nnue_path = Path(__file__).parent.parent.parent / "sunfish_nnue.py"
    
    for model_file in model_files:
        if not Path(model_file).exists():
            print(f"{Path(model_file).name:25} Not found")
            continue
            
        print(f"{Path(model_file).name:25} ", end="", flush=True)
        
        total_time = 0
        move_count = 0
        
        try:
            for i, fen in enumerate(test_positions):
                # Prepare UCI commands
                commands = [
                    "uci",
                    "isready",
                    f"position fen {fen}",
                    f"go depth {depth}",
                    "quit"
                ]
                
                start_time = time.time()
                
                process = subprocess.Popen(
                    [sys.executable, str(sunfish_nnue_path), str(model_file)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                commands_str = "\n".join(commands) + "\n"
                stdout, stderr = process.communicate(input=commands_str, timeout=60)
                
                elapsed = time.time() - start_time
                total_time += elapsed
                
                if "bestmove" in stdout:
                    move_count += 1
                
                print(".", end="", flush=True)
            
            avg_time = total_time / len(test_positions)
            print(f" {avg_time:.2f}s avg, {move_count}/{len(test_positions)} moves")
            
        except Exception as e:
            print(f" Error: {str(e)[:20]}...")

def main():
    parser = argparse.ArgumentParser(description="Evaluate NNUE models")
    parser.add_argument("--models", nargs="+", help="Model files to evaluate")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument("--depth", type=int, default=4, help="Search depth for benchmark")
    parser.add_argument("--positions", type=int, default=5, help="Number of test positions")
    
    args = parser.parse_args()
    
    if not args.models:
        # Default: look for models in current directory
        model_dir = Path("models")
        if model_dir.exists():
            args.models = list(model_dir.glob("*.pickle"))
        else:
            print("No models specified and no models/ directory found")
            return
    
    print("NNUE Model Evaluation")
    print("=" * 50)
    
    # Test compatibility
    for model_file in args.models:
        test_model_compatibility(model_file)
        print()
    
    # Compare sizes
    compare_model_sizes(args.models)
    
    # Benchmark performance
    if args.benchmark:
        benchmark_models(args.models, args.depth, args.positions)

if __name__ == "__main__":
    main()