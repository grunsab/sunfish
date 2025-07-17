# Sunfish NNUE Training Pipeline

This directory contains a complete training pipeline for NNUE (Efficiently Updatable Neural Networks) models compatible with the Sunfish chess engine.

## Overview

The training pipeline creates neural network models that can be used directly with `sunfish_nnue.py` without any modifications to the engine code. The models are trained on the CCRL (Computer Chess Rating Lists) dataset, which contains high-quality computer chess games.

## Features

- **Compatible Format**: Models export to the exact pickle format expected by `sunfish_nnue.py`
- **Optimized for Size**: Focus on creating minimal models (target: <3KB) 
- **CCRL Dataset**: Uses professional computer chess games for training
- **Multiple Model Sizes**: Configurable architectures (tiny/small/medium/large)
- **Training Monitoring**: Loss plots and validation tracking
- **Model Evaluation**: Compatibility testing and performance benchmarking

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download CCRL Data

```bash
python download_ccrl.py
```

Follow the instructions to manually download CCRL PGN files from:
- https://www.computerchess.org.uk/ccrl/4040/games.html
- https://computerchess.org.uk/ccrl/404/games.html

Save PGN files in the `data/` directory.

### 3. Preprocess Data

```bash
python preprocess_data.py
```

This extracts training positions from PGN files and creates JSON datasets.

### 4. Train Model

```bash
python train.py --train-data processed_data/train_data.json --val-data processed_data/val_data.json --model-size small
```

### 5. Test Model

```bash
python evaluate_model.py --models models/best_model_small.pickle
```

### 6. Use with Sunfish

```bash
# Test the new model
tools/fancy.py -cmd "./sunfish_nnue.py nnue/training/models/best_model_small.pickle"
```

## Architecture

### NNUE Model Structure

The model architecture matches the existing `sunfish_nnue.py` format:

```
Position Embedding (8×8×6) 
    ↓
Piece Combination (L0×6×6)
    ↓  
Color Combination (L0×L0×2)
    ↓
Features: [piece_balance, position_features...]
    ↓
Hidden Layer 1 (2×(L0-1) → L2) + tanh
    ↓
Hidden Layer 2 (L2 → 1) + piece_balance_scaling
    ↓
Final Evaluation
```

### Model Sizes

| Size   | L0/L1/L2 | Approx. Size | Use Case |
|--------|----------|--------------|----------|
| tiny   | 6×6×6    | ~1KB         | Ultra-minimal |
| small  | 8×8×8    | ~2KB         | Recommended |
| medium | 10×10×10 | ~3KB         | Default sunfish size |
| large  | 16×16×16 | ~8KB         | Maximum quality |

## File Structure

```
nnue/training/
├── README.md                 # This documentation
├── requirements.txt          # Python dependencies
├── download_ccrl.py         # CCRL dataset download helper
├── preprocess_data.py       # PGN to training data conversion
├── nnue_model.py           # PyTorch NNUE model implementation
├── train.py                # Main training script
├── evaluate_model.py       # Model testing and benchmarking
├── data/                   # CCRL PGN files (gitignored)
├── processed_data/         # Preprocessed JSON datasets (gitignored)
└── models/                 # Trained models
```

## Detailed Usage

### Data Preprocessing Options

```bash
# Process with custom limits
python preprocess_data.py --max-games 100000 --max-positions-per-game 30

# Extract only decisive games (no draws)
python preprocess_data.py --decisive-only
```

### Training Options

```bash
# Train tiny model for quick testing
python train.py --model-size tiny --epochs 50 --batch-size 512

# Train with GPU acceleration
python train.py --device cuda --batch-size 1024

# Resume from checkpoint
python train.py --resume models/checkpoint_small_epoch_50.pth

# Train with limited data for testing
python train.py --max-train-samples 10000 --max-val-samples 2000
```

### Model Evaluation

```bash
# Test multiple models
python evaluate_model.py --models models/*.pickle

# Run performance benchmark
python evaluate_model.py --benchmark --depth 5 --positions 10

# Compare with existing sunfish models
python evaluate_model.py --models models/best_model_small.pickle ../models/tanh.pickle
```

## Training Tips

### Model Size Selection

- **tiny**: For experimentation and ultra-fast training
- **small**: Recommended for production use (good size/strength balance)
- **medium**: Matches existing sunfish NNUE models
- **large**: Maximum strength (but may be slower)

### Training Parameters

- **Learning Rate**: Start with 0.001, reduce if loss plateaus
- **Batch Size**: Larger is generally better (256-1024)
- **Epochs**: Early stopping typically occurs around 50-100 epochs
- **Data Size**: More positions = better model (target: 100K+ positions)

### Data Quality

- Focus on games from engines rated 2800+ Elo
- Include diverse openings and time controls
- Balance decisive games vs. draws (roughly 60/40 split)
- Avoid positions from the first 10 and last 10 moves

## Model Format Compatibility

The exported models use the exact format expected by `sunfish_nnue.py`:

```python
model = {
    "ars": [array0, array1, array2, array3, array4, array5],  # int8 arrays
    "scale": float_value  # piece balance scaling factor
}
```

Arrays contain:
0. Position embedding (8×8×6)
1. Piece combination (L0×6×6) 
2. Piece values (6)
3. Color combination (L0×L0×2)
4. Layer 1 weights (L2×2×(L0-1))
5. Layer 2 weights (1×L2)

## Performance Optimization

### For Training Speed

- Use smaller model sizes for experimentation
- Reduce dataset size with `--max-train-samples`
- Increase batch size if memory allows
- Use GPU acceleration with `--device cuda`

### For Model Size

- Use "tiny" or "small" configurations
- The model automatically uses int8 quantization
- Final models should be under 3KB for fast loading

### For Model Strength

- Train on more diverse, high-quality positions
- Use larger model architectures
- Train for more epochs with early stopping
- Balance training data across game phases

## Troubleshooting

### Common Issues

**ModuleNotFoundError: No module named 'chess'**
```bash
pip install python-chess
```

**Model compatibility test fails**
- Check model file exists and is valid pickle format
- Verify sunfish_nnue.py path is correct
- Test with a known working model first

**Training is very slow**
- Reduce batch size or model size
- Use fewer training samples for testing
- Consider GPU acceleration

**Model performs poorly**
- Check data preprocessing quality
- Ensure sufficient training data diversity
- Verify evaluation function in preprocessing
- Try different model architectures

### Validation

Always validate new models before deployment:

```bash
# 1. Compatibility test
python evaluate_model.py --models models/new_model.pickle

# 2. Quick game test
tools/fancy.py -cmd "./sunfish_nnue.py nnue/training/models/new_model.pickle"

# 3. Engine match (if available)
tools/test.sh  # With new model in nnue/models/
```

## Contributing

When contributing new features:

1. Maintain compatibility with existing `sunfish_nnue.py`
2. Keep model export format exactly consistent
3. Add tests for new functionality
4. Update documentation for new options
5. Verify models are minimal in size

## License

This training pipeline follows the same license as the Sunfish project (GPL v3).