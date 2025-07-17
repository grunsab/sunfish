# Sunfish Policy Network Integration

This repository implements a policy network integration for the Sunfish chess engine, where **the policy equals the next move that's played**. The implementation is based on the [pytorch-alpha-zero](https://github.com/grunsab/pytorch-alpha-zero) repository but adapted for Sunfish's unique architecture.

## 🎯 Core Concept

The key innovation is that the policy network learns directly from played moves:
- **During search**: The policy network guides move ordering for better search efficiency
- **During play**: Every move played becomes a training example for the policy network
- **Over time**: The policy network gradually learns to predict the moves that strong players would make

## 📁 Files Overview

### Core Implementation
- `sunfish_policy_integrated.py` - Main engine with policy network integration
- `nnue/training/policy_network.py` - Policy network architecture and training infrastructure
- `nnue/training/train_policy.py` - Training script for policy network
- `policy_example.py` - Usage examples and demonstrations

### Architecture

The policy network uses a ResNet-like architecture similar to AlphaZero:

```
Input (16 channels, 8x8 board) 
    ↓
ConvBlock (initial convolution + batch norm + ReLU)
    ↓
ResidualBlocks (4 blocks with skip connections)
    ↓
PolicyHead (2 conv channels → 4096 move outputs)
    ↓
Move Probabilities (softmax over legal moves)
```

## 🚀 Quick Start

### 1. Basic Usage

```bash
# Run the policy-integrated engine
python3 sunfish_policy_integrated.py

# Or with existing model
python3 sunfish_policy_integrated.py --policy-model my_model.pth
```

### 2. Play Against the Engine

```bash
# Using the fancy terminal interface
python3 tools/fancy.py -cmd "./sunfish_policy_integrated.py"
```

### 3. Run Examples

```bash
# See the policy network in action
python3 policy_example.py
```

## 🎓 Training the Policy Network

### Step 1: Create Training Data

```bash
cd nnue/training

# Create training data from CCRL games
python3 train_policy.py --create-data --data-dir ../../../data
```

### Step 2: Train the Network

```bash
# Train a small model
python3 train_policy.py --train-data-json policy_training_data.json --model-size small

# Train with GPU acceleration
python3 train_policy.py --train-data-json policy_training_data.json --device cuda --batch-size 64
```

### Step 3: Use the Trained Model

```bash
# Run engine with trained model
python3 sunfish_policy_integrated.py --policy-model policy_model_small_final.pth
```

## 📊 How It Works

### 1. Policy-Guided Search

The search algorithm uses the policy network to order moves:

```python
# Get policy probabilities for legal moves
if policy_manager and policy_manager.enabled:
    policy_result = policy_manager.get_move_probabilities(pos.board, legal_moves)
    if policy_result:
        move_probs, _ = policy_result
        # Sort moves by policy probability (descending)
        sorted_moves = sorted(legal_moves, key=lambda m: move_probs.get(m, 0), reverse=True)
```

### 2. Learning from Played Moves

Every move played becomes a training example:

```python
# Update policy network with the move we played
if best_move and policy_manager:
    policy_manager.update_policy(hist[-1].board, best_move, game_result)
```

### 3. Continuous Learning

The policy network trains incrementally:
- Stores recent positions and moves
- Periodically trains on accumulated examples
- Gradually improves move prediction accuracy

## 🔧 Configuration Options

### Model Sizes

| Size   | Blocks | Channels | Parameters | Use Case |
|--------|--------|----------|------------|----------|
| tiny   | 2      | 64       | ~50K       | Testing |
| small  | 4      | 128      | ~200K      | Recommended |
| medium | 6      | 256      | ~800K      | High quality |
| large  | 8      | 512      | ~3M        | Maximum strength |

### Command Line Options

```bash
# Policy model options
--policy-model PATH     # Load existing model
--policy-size SIZE      # Model size (tiny/small/medium/large)

# Training options
--create-data           # Create training data from CCRL
--train-data-json FILE  # Train on existing data
--epochs N             # Number of training epochs
--batch-size N         # Training batch size
--learning-rate F      # Learning rate
--device cpu/cuda      # Device to use
```

## 🧪 Testing and Evaluation

### Run Basic Tests

```bash
# Test original engine
bash tools/quick_tests.sh ./sunfish.py

# Test policy-integrated engine (requires removing policy for compatibility)
bash tools/quick_tests.sh ./sunfish_policy_integrated.py
```

### Performance Evaluation

```bash
# Test network functionality
python3 -c "
import sys
sys.path.append('nnue/training')
from policy_network import test_policy_network
test_policy_network()
"
```

## 📈 Training Data

The system uses CCRL (Computer Chess Rating Lists) games for training:

### Data Structure
```
data/
├── cclr/
│   ├── train/        # Training games (250 PGN files)
│   └── test/         # Test games (250 PGN files)
```

### Processing Pipeline
1. **Parse PGN**: Extract games and moves
2. **Convert Format**: Map to Sunfish board representation
3. **Create Examples**: Position + move_played pairs
4. **Train Network**: Learn to predict played moves

## 🔬 Technical Details

### Board Representation

The policy network uses a 16-channel 8x8 tensor:
- Channels 0-5: White pieces (P, N, B, R, Q, K)
- Channels 6-11: Black pieces (p, n, b, r, q, k)
- Channels 12-15: Game state (castling rights, en passant, etc.)

### Move Encoding

Moves are encoded as indices into a 4096-dimensional output:
- Regular moves: from_square * 64 + to_square
- Promotions: Additional indices for each promotion piece
- Total: ~20,000 possible moves mapped to 4096 outputs

### Training Process

The policy network learns through supervised learning:
1. **Input**: Board position (16×8×8 tensor)
2. **Target**: Move that was actually played (one-hot encoded)
3. **Loss**: Cross-entropy between predicted and actual move
4. **Optimization**: Adam optimizer with learning rate decay

## 🤝 Integration with Existing Code

The policy integration is designed to be non-intrusive:

### Backward Compatibility
- Works with existing UCI protocol
- Falls back gracefully when PyTorch unavailable
- Maintains same move generation and evaluation logic

### Modular Design
- `PolicyManager` handles all policy-related operations
- Search algorithm enhanced but not fundamentally changed
- Training infrastructure separate from engine code

## 🎯 Performance Characteristics

### Search Improvement
- Better move ordering leads to more efficient search
- Reduces average branching factor
- Finds good moves earlier in the search tree

### Learning Behavior
- Gradual improvement over many games
- Learns opening principles and tactical patterns
- Adapts to the playing style of opponents

### Resource Usage
- Minimal overhead when policy network disabled
- GPU acceleration optional but recommended for training
- Model size configurable based on requirements

## 🔮 Future Enhancements

### Potential Improvements
1. **Value Network**: Add position evaluation component
2. **MCTS Integration**: Implement Monte Carlo Tree Search
3. **Self-Play Training**: Generate training data through self-play
4. **Opening Book**: Integrate with policy network
5. **Endgame Tablebase**: Combine with policy guidance

### Research Directions
1. **Architecture Optimization**: Experiment with different network structures
2. **Training Efficiency**: Investigate better training procedures
3. **Transfer Learning**: Apply knowledge from other chess engines
4. **Distributed Training**: Scale to larger datasets

## 📚 References

- [pytorch-alpha-zero](https://github.com/grunsab/pytorch-alpha-zero) - Original inspiration
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815) - Mastering Chess and Shogi by Self-Play
- [Sunfish Engine](https://github.com/thomasahle/sunfish) - Original engine implementation
- [CCRL Database](https://www.computerchess.org.uk/ccrl/) - Computer Chess Rating Lists

## 🐛 Troubleshooting

### Common Issues

**Policy network not loading**
- Check PyTorch installation: `pip install torch`
- Verify model file exists and is readable
- Ensure compatible Python version (3.7+)

**Training fails**
- Check CCRL data directory structure
- Verify python-chess installation: `pip install python-chess`
- Reduce batch size if memory issues occur

**Poor performance**
- Try smaller model size for faster inference
- Increase training data size for better accuracy
- Experiment with different learning rates

### Debug Mode

Enable debug output:
```bash
python3 sunfish_policy_integrated.py --debug
```

## 📄 License

This implementation follows the same license as the original Sunfish project (GPL v3).

## 🤗 Contributing

Contributions are welcome! Please:
1. Maintain compatibility with existing Sunfish code
2. Add tests for new functionality
3. Update documentation for new features
4. Follow the existing code style

---

*This policy network integration demonstrates how modern deep learning techniques can enhance traditional chess engines while maintaining their essential characteristics.*