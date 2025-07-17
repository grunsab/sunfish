#!/usr/bin/env python3
"""Test script to check if imports are working correctly"""

import sys
from pathlib import Path

print("Testing imports...")

# Test basic imports
try:
    import argparse
    print("✓ argparse imported")
except ImportError as e:
    print(f"✗ Failed to import argparse: {e}")

try:
    import os
    print("✓ os imported")
except ImportError as e:
    print(f"✗ Failed to import os: {e}")

try:
    import json
    print("✓ json imported")
except ImportError as e:
    print(f"✗ Failed to import json: {e}")

try:
    import time
    print("✓ time imported")
except ImportError as e:
    print(f"✗ Failed to import time: {e}")

# Test external dependencies
try:
    import numpy as np
    print("✓ numpy imported")
except ImportError as e:
    print(f"✗ Failed to import numpy: {e}")

try:
    import torch
    print(f"✓ torch imported (version: {torch.__version__})")
except ImportError as e:
    print(f"✗ Failed to import torch: {e}")

try:
    import torch.nn as nn
    print("✓ torch.nn imported")
except ImportError as e:
    print(f"✗ Failed to import torch.nn: {e}")

try:
    from torch.utils.data import Dataset, DataLoader
    print("✓ torch.utils.data imported")
except ImportError as e:
    print(f"✗ Failed to import torch.utils.data: {e}")

try:
    from tqdm import tqdm
    print("✓ tqdm imported")
except ImportError as e:
    print(f"✗ Failed to import tqdm: {e}")

try:
    import chess
    print(f"✓ chess imported (version: {chess.__version__})")
except ImportError as e:
    print(f"✗ Failed to import chess: {e}")

try:
    import chess.pgn
    print("✓ chess.pgn imported")
except ImportError as e:
    print(f"✗ Failed to import chess.pgn: {e}")

# Test local imports
print("\nTesting local imports...")
sys.path.append(str(Path(__file__).parent.parent.parent))

try:
    from policy_network import SunfishPolicyNetwork, PolicyTrainer, create_policy_network
    print("✓ policy_network imported")
except ImportError as e:
    print(f"✗ Failed to import policy_network: {e}")

try:
    from sunfish import Position, Move, initial
    print("✓ sunfish imported")
except ImportError as e:
    print(f"✗ Failed to import sunfish: {e}")

print("\nAll import tests completed!")