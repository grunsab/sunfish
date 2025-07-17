#!/bin/bash

# Example training script for Sunfish NNUE models
# This script demonstrates the full training pipeline

echo "Sunfish NNUE Training Pipeline"
echo "==============================="

# Check if we're in the right directory
if [ ! -f "nnue_model.py" ]; then
    echo "Error: Please run this script from the nnue/training/ directory"
    exit 1
fi

# Step 1: Download CCRL data (manual step)
echo "Step 1: Download CCRL Data"
echo "Please run: python download_ccrl.py"
echo "And follow the instructions to download PGN files"
echo ""

# Check if data directory exists
if [ ! -d "data" ]; then
    echo "Creating data directory..."
    mkdir -p data
fi

# Step 2: Check for PGN files
echo "Step 2: Checking for PGN files..."
pgn_count=$(find data -name "*.pgn" | wc -l)
if [ $pgn_count -eq 0 ]; then
    echo "No PGN files found in data/ directory"
    echo "Please download CCRL PGN files first"
    exit 1
else
    echo "Found $pgn_count PGN files"
fi

# Step 3: Preprocess data
echo "Step 3: Preprocessing data..."
if [ ! -f "processed_data/train_data.json" ]; then
    echo "Running data preprocessing..."
    python preprocess_data.py
    if [ $? -ne 0 ]; then
        echo "Preprocessing failed!"
        exit 1
    fi
else
    echo "Preprocessed data already exists"
fi

# Step 4: Train models
echo "Step 4: Training models..."

# Create models directory
mkdir -p models

# Train small model (recommended)
echo "Training small model..."
python train.py \
    --train-data processed_data/train_data.json \
    --val-data processed_data/val_data.json \
    --model-size small \
    --epochs 100 \
    --batch-size 256 \
    --learning-rate 0.001 \
    --save-dir models

# Train tiny model for ultra-fast loading
echo "Training tiny model..."
python train.py \
    --train-data processed_data/train_data.json \
    --val-data processed_data/val_data.json \
    --model-size tiny \
    --epochs 50 \
    --batch-size 512 \
    --learning-rate 0.001 \
    --save-dir models

# Step 5: Evaluate models
echo "Step 5: Evaluating models..."
python evaluate_model.py --models models/*.pickle

# Step 6: Test with sunfish
echo "Step 6: Testing with sunfish..."
if [ -f "models/best_model_small.pickle" ]; then
    echo "Testing small model with sunfish_nnue.py..."
    # This is just a compatibility test
    echo "uci" | python ../../../sunfish_nnue.py models/best_model_small.pickle
    if [ $? -eq 0 ]; then
        echo "✓ Small model is compatible!"
    else
        echo "✗ Small model failed compatibility test"
    fi
fi

echo ""
echo "Training pipeline completed!"
echo "Models saved in: models/"
echo "Use with: tools/fancy.py -cmd \"./sunfish_nnue.py nnue/training/models/best_model_small.pickle\""