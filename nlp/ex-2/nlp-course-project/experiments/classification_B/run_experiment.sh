#!/bin/bash
# Script to run the full RNN classification experiment pipeline (Experiment B)

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

echo "Starting Classification Experiment B: RNN with Word2Vec Embeddings"
echo "=================================================================="

# Step 1: Set up the RNN classification model
echo "Step 1: Setting up the RNN classification model..."
python3 "$SCRIPT_DIR/setup_classification_model.py" \
    --model-dir "$MODEL_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --embedding-dim 300 \
    --hidden-dim 256 \
    --num-layers 2 \
    --dropout 0.3 \
    --rnn-type lstm \
    --bidirectional \
    --min-freq 2

echo "Setup complete. Configuration saved in $OUTPUT_DIR"
echo "------------------------------------"

# Step 2: Train the RNN classification model
echo "Step 2: Training the RNN classification model..."
python3 "$SCRIPT_DIR/train_classification.py" \
    --model-dir "$MODEL_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --config-file "$OUTPUT_DIR/rnn_classifier_config.json" \
    --epochs 3 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --max-length 200

echo "Training complete. Trained model saved in $OUTPUT_DIR"
echo "------------------------------------"

# Step 3: Evaluate the trained model
echo "Step 3: Evaluating the trained model..."
python3 "$SCRIPT_DIR/evaluate_classification.py" \
    --model-dir "$MODEL_DIR" \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --trained-model-file "trained_rnn_classifier.pth" \
    --config-file "$OUTPUT_DIR/rnn_classifier_config.json"

echo "Evaluation complete. Results and plots saved in $OUTPUT_DIR"
echo "------------------------------------"

echo "Experiment B finished successfully!"
echo ""
echo "Summary of Experiment B:"
echo "- Model: From-scratch RNN (LSTM) with pre-trained Word2Vec embeddings"
echo "- Approach: End-to-end training of RNN classifier"
echo "- Key features:"
echo "  * Pre-trained Word2Vec embeddings (300D)"
echo "  * Bidirectional LSTM (2 layers, 256 hidden dim)"
echo "  * Dropout regularization (0.3)"
echo "  * Adam optimizer with learning rate scheduling"
echo "  * Early stopping to prevent overfitting"
echo ""
echo "Results saved in: $OUTPUT_DIR" 