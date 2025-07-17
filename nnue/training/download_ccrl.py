#!/usr/bin/env python3
"""
Download CCRL dataset for NNUE training.
CCRL (Computer Chess Rating Lists) provides high-quality computer chess games.
"""

import os
import sys
import urllib.request
import urllib.parse
from pathlib import Path
import zipfile
import gzip

def download_ccrl_games(data_dir="data"):
    """Download CCRL games in PGN format."""
    
    # Create data directory
    Path(data_dir).mkdir(exist_ok=True)
    
    # CCRL download URLs - these are example URLs, the actual ones may vary
    # The user will need to manually download from:
    # https://www.computerchess.org.uk/ccrl/4040/games.html
    # https://computerchess.org.uk/ccrl/404/games.html
    
    print("=" * 60)
    print("CCRL Dataset Download Instructions")
    print("=" * 60)
    print()
    print("This script helps prepare for CCRL dataset download.")
    print("You need to manually download PGN files from:")
    print()
    print("1. CCRL 40/15: https://www.computerchess.org.uk/ccrl/4040/games.html")
    print("2. CCRL 40/4:  https://computerchess.org.uk/ccrl/404/games.html")
    print()
    print("Download instructions:")
    print("- Download monthly PGN files or engine-specific files")
    print("- Save them in the '{}' directory".format(data_dir))
    print("- Files should be named like: ccrl_202X_XX.pgn")
    print()
    print("Example downloads:")
    print("- Recent months for diverse games")
    print("- Focus on engines rated 2800+ for quality")
    print()
    
    # Check if any PGN files already exist
    pgn_files = list(Path(data_dir).glob("*.pgn"))
    if pgn_files:
        print("Found existing PGN files:")
        for f in pgn_files:
            size_mb = f.stat().st_size / (1024*1024)
            print(f"  - {f.name} ({size_mb:.1f} MB)")
        print()
    else:
        print("No PGN files found in '{}' directory.".format(data_dir))
        print("Please download some CCRL PGN files first.")
        print()
    
    return len(pgn_files) > 0

if __name__ == "__main__":
    success = download_ccrl_games()
    if success:
        print("Ready to process CCRL data!")
    else:
        print("Please download CCRL PGN files first.")
        sys.exit(1)