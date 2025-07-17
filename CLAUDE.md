# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Sunfish is a simple, strong chess engine written in Python. The core engine is designed to be minimal (131 lines when cleaned) while still playing at 2000+ rating on Lichess. It uses the UCI protocol and includes both a standard version and an experimental NNUE (neural network) version.

## Key Commands

### Running Sunfish

- **Interactive play**: `tools/fancy.py -cmd ./sunfish.py` (requires python-chess)
- **NNUE version**: `tools/fancy.py -cmd "./sunfish_nnue.py nnue/models/tanh.pickle"`
- **Direct UCI mode**: `./sunfish.py` (communicate via UCI protocol)
- **Packed version**: `build/pack.sh sunfish.py packed.sh` (creates <3KB executable)

### Testing

- **Quick tests**: `tools/quick_tests.sh sunfish.py`
- **Full test suite**: `tools/test.sh` (uses cutechess-cli for engine vs engine matches)
- **Specific tests via tester.py**:
  - Mate puzzles: `python3 tools/tester.py sunfish.py mate tools/test_files/mate1.fen --movetime 10000`
  - Perft: `python3 tools/tester.py sunfish.py perft tools/test_files/perft.epd --depth 2`
  - Self-play: `python3 tools/tester.py sunfish.py self-play --time 1000 --inc 100`

### Building

- **Clean/minify**: `build/clean.sh sunfish.py` (removes comments and whitespace)
- **Pack executable**: `build/pack.sh sunfish.py output.sh` (creates compressed standalone executable)

## Architecture

### Core Components

1. **sunfish.py**: Main chess engine implementation
   - MTD-bi search algorithm (C*)
   - Piece-square tables for evaluation
   - Board representation using Python strings
   - UCI protocol implementation

2. **sunfish_nnue.py**: Neural network enhanced version
   - Uses small NNUE models (1207 bytes)
   - Better positional play but slower tactics
   - Models stored in `nnue/models/`

3. **tools/fancy.py**: Terminal interface for human play
   - Supports algebraic notation
   - Visual board display with Unicode pieces
   - Wrapper around UCI engines

4. **tools/tester.py**: Comprehensive testing framework
   - Mate puzzle solving
   - Perft validation
   - EPD test suites
   - Self-play capabilities

### Board Representation

- Uses string-based board representation for simplicity
- Board is a 120-character string (10x12 with padding)
- Uppercase = white pieces, lowercase = black pieces
- Empty squares = '.'
- Padding squares for boundary detection

### Search Algorithm

- MTD-bi (Memory-enhanced Test Driver with binary search)
- Null move pruning
- Quiescence search
- No dedicated check detection or evasion (kept simple)

## Development Notes

### Performance

- Best run with PyPy3 for ~250 ELO improvement
- PyPy2.7 gives best performance at fast time controls
- Standard CPython is significantly slower

### Code Style

- Prioritize simplicity and readability
- Minimize lines of code while maintaining functionality
- Use standard Python collections and idioms
- Piece values and PST tables are the main tuning parameters

### Testing Approach

- Use `tools/tester.py` for automated testing
- Test files in `tools/test_files/` include:
  - Mate puzzles (mate1.fen through mate4.fen)
  - Perft positions
  - EPD test suites (Bratko-Kopec, CCR, Win at Chess)
  - Stalemate positions
  - Game collections for opening tests

### Limitations

- No 50-move draw rule implementation
- No bitboards (uses string representation)
- No specialized move generation
- Limited endgame knowledge