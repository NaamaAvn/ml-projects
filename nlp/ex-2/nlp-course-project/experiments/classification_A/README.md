# Classification Experiment A: Pre-trained Language Model as Feature Extractor

## Overview

This experiment implements a **feature extraction approach** for text classification using a **pre-trained language model** as a backbone. The language model encodes input text into sentence embeddings, which are then fed into a separate classification head (MLP) for binary classification on the IMDB dataset.

### Key Features
- **Feature extraction approach**: Pre-trained language model weights remain frozen
- **Efficient training**: Only the classification head is trained
- **Multiple pooling methods**: Mean, max, or last hidden state pooling
- **Lightweight architecture**: Small classification head on top of frozen backbone
- **Fast inference**: Leverages pre-trained knowledge without full retraining

### Comparison with Experiment B

| Aspect | Experiment A | Experiment B |
|--------|-------------|--------------|
| **Model Architecture** | Pre-trained LM + MLP classifier | From-scratch RNN + classification head |
| **Embeddings** | Learned during LM pre-training | Pre-trained Word2Vec (Google News 300D) |
| **Training Approach** | Feature extraction (frozen LM) | End-to-end training |
| **Model Complexity** | Two-stage: LM + classifier | Single-stage: RNN classifier |
| **Computational Cost** | Lower (frozen backbone) | Higher (full training) |
| **Flexibility** | Limited by pre-trained LM | Fully customizable |

## Model Architecture

### Two-Stage Classification Model

1. **Pre-trained Language Model (Frozen Backbone)**
   - **Architecture**: LSTM-based language model
   - **Embedding dimension**: 128
   - **Hidden dimension**: 256
   - **Number of layers**: 2
   - **Status**: Frozen (no gradient updates during training)
   - **Purpose**: Extract sentence-level representations

2. **Feature Extractor**
   - **Input**: Tokenized text sequences
   - **Output**: Sentence embeddings (256D)
   - **Pooling methods**: 
     - **Mean pooling**: Average of all non-padding hidden states
     - **Max pooling**: Maximum value across sequence dimension
     - **Last pooling**: Final hidden state from top LSTM layer
   - **Padding handling**: Masks padding tokens during pooling

3. **Classification Head (MLP)**
   - **Input**: Sentence embeddings (256D)
   - **Hidden layers**: 256D → 128D (configurable)
   - **Output**: 2D (binary classification: negative/positive)
   - **Activation**: ReLU between layers
   - **Regularization**: Dropout (0.3) for generalization

### Key Implementation Features
- **Frozen backbone**: Language model parameters remain unchanged
- **Efficient pooling**: Handles variable-length sequences with padding masks
- **Gradient clipping**: Prevents exploding gradients during training
- **Early stopping**: Prevents overfitting based on validation performance
- **Learning rate scheduling**: ReduceLROnPlateau scheduler for adaptive learning rate
- **Flexible architecture**: Configurable hidden dimensions and pooling methods

## File Structure

```
classification_A/
├── setup_classification_model.py    # Model setup and configuration
├── train_classification.py          # Training script
├── evaluate_classification.py       # Evaluation and analysis
├── run_experiment.sh               # Complete experiment pipeline
├── README.md                       # This file
└── results/                        # Output directory (created during execution)
    ├── classification_model_config.json  # Model configuration
    ├── trained_classification_model.pth  # Trained model weights (~37MB)
    ├── training_history.png        # Training metrics visualization
    ├── confusion_matrix.png        # Confusion matrix plot
    ├── classification_test_results.json  # Test set evaluation metrics
    └── error_analysis.json         # Detailed error analysis (~11MB)
```

## Usage

### Quick Start
Run the complete experiment pipeline:
```bash
./run_experiment.sh
```

### Step-by-Step Execution

1. **Setup Model**:
   ```bash
   python3 setup_classification_model.py \
       --model-dir ../../model/ \
       --data-dir ../../data/processed_classification_data/ \
       --output-dir ./results \
       --pooling-method mean \
       --hidden-dims 256 128 \
       --dropout 0.3
   ```

2. **Train Model**:
   ```bash
   python3 train_classification.py \
       --model-dir ../../model/ \
       --data-dir ../../data/processed_classification_data/ \
       --output-dir ./results \
       --config-file ./results/classification_model_config.json \
       --epochs 10 \
       --batch-size 32 \
       --learning-rate 0.001 \
       --max-length 200
   ```

3. **Evaluate Model**:
   ```bash
   python3 evaluate_classification.py \
       --model-dir ../../model/ \
       --data-dir ../../data/processed_classification_data/ \
       --output-dir ./results \
       --trained-model-file trained_classification_model.pth \
       --max-length 200
   ```

## Configuration Options

### Model Parameters
- `--pooling-method`: Sentence embedding pooling method ('mean', 'max', 'last')
- `--hidden-dims`: List of hidden layer dimensions for classification head
- `--dropout`: Dropout rate for classification head (default: 0.3)
- `--feature-dim`: Dimension of sentence embeddings (from language model)

### Training Parameters
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size for training (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--max-length`: Maximum sequence length (default: 200)

### Data Parameters
- `--model-dir`: Directory containing pre-trained language model
- `--data-dir`: Directory containing processed classification data
- `--output-dir`: Directory to save results and models
- `--device`: Device to run training on ('cpu' or 'cuda')

## Dependencies

### Core Dependencies
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **Matplotlib**: Plotting and visualization
- **Scikit-learn**: Machine learning metrics and utilities

### Specialized Dependencies
- **tqdm**: Progress bars for training loops
- **JSON**: Configuration and data serialization

## Notes

### Performance Expectations
- **Training Time**: ~10-20 minutes on CPU, ~2-5 minutes on GPU
- **Memory Usage**: ~2GB RAM (lightweight compared to Experiment B)
- **Model Size**: ~37MB for trained classification head
- **Vocabulary Size**: ~49K tokens (shared with pre-trained language model)
- **Test Accuracy**: ~66.8% (based on actual results)

### Pre-trained Language Model Requirements
- **Model file**: `language_model.pth` must exist in `../../model/`
- **Vocabulary**: Shared vocabulary from language model training (`../../data/vocab.json`)
- **Architecture compatibility**: Classification head designed for 256D embeddings
- **Frozen parameters**: Language model weights remain unchanged during training

### Training Features
- **Feature extraction**: Only classification head parameters are updated
- **Gradient clipping**: Prevents exploding gradients (max_norm=1.0)
- **Early stopping**: Prevents overfitting based on validation accuracy
- **Learning rate scheduling**: ReduceLROnPlateau with factor=0.5, patience=2
- **Regularization**: Dropout in classification head for generalization

### Output Files
- **Model Files**: Trained classification head weights and configuration
- **Visualizations**: Training curves and confusion matrix
- **Metrics**: Detailed evaluation reports and classification metrics
- **Error Analysis**: Comprehensive analysis of misclassified examples

### Advantages
- **Fast training**: Only small classification head needs training
- **Leverages pre-trained knowledge**: Uses language model's learned representations
- **Memory efficient**: Smaller model size and memory footprint
- **Stable training**: Frozen backbone provides consistent features

### Limitations
- **Limited flexibility**: Architecture constrained by pre-trained language model
- **Feature dependency**: Performance depends on quality of pre-trained features
- **Domain mismatch**: Pre-trained model may not be optimal for target domain
- **Moderate accuracy**: ~67% test accuracy compared to state-of-the-art models 