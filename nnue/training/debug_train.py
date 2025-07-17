#!/usr/bin/env python3
"""Debug script to test where train_policy.py is hanging"""

import sys
import os

print("Debug script starting...", flush=True)

# Change to the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Working directory: {os.getcwd()}", flush=True)

# Test command line parsing
print("\nTesting with arguments: --create-data --data-dir ../../../data", flush=True)

# Run the script with minimal debugging
try:
    # Set command line arguments
    sys.argv = ['train_policy.py', '--create-data', '--data-dir', '../../../data']
    
    print("About to import train_policy...", flush=True)
    import train_policy
    print("Import successful!", flush=True)
    
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("Debug script complete", flush=True)