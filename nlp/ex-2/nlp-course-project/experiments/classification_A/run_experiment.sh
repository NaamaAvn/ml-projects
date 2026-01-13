#!/bin/bash
# Script to run the full classification experiment pipeline

# Exit immediately if a command exits with a non-zero status.
set -e

# Get the absolute path of the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Define directories relative to the script's location
OUTPUT_DIR="$SCRIPT_DIR/results"
MODEL_DIR="$SCRIPT_DIR/../../model"
DATA_DIR="$SCRIPT_DIR/../../data/processed_classification_data"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Starting Classification Experiment A"
echo "===================================="

# Step 1: Set up the classification model
echo "Step 1: Setting up the classification model..."
python3 "$SCRIPT_DIR/setup_classification_model.py" \
    --model-dir "$MODEL_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR"

echo "Setup complete. Configuration saved in $OUTPUT_DIR"
echo "------------------------------------"

# Step 2: Train the classification model
echo "Step 2: Training the classification model..."
python3 "$SCRIPT_DIR/train_classification.py" \
    --model-dir "$MODEL_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --config-file "$OUTPUT_DIR/classification_model_config.json" \
    --epochs 5 # Using a small number of epochs for a quick run

echo "Training complete. Trained model saved in $OUTPUT_DIR"
echo "------------------------------------"

# Step 3: Evaluate the trained model
echo "Step 3: Evaluating the trained model..."
python3 "$SCRIPT_DIR/evaluate_classification.py" \
    --model-dir "$MODEL_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --trained-model-file "trained_classification_model.pth"

echo "Evaluation complete. Results and plots saved in $OUTPUT_DIR"
echo "------------------------------------"

echo "Experiment finished successfully!" 