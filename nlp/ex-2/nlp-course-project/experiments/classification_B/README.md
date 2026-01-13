# Classification Experiment B: RNN with Word2Vec Embeddings

## Overview

This experiment implements a **from-scratch RNN classifier** using **pre-trained Word2Vec embeddings** for text classification on the IMDB dataset. This approach provides an alternative to Experiment A by training a complete neural network from scratch rather than using a pre-trained language model as a feature extractor.

### Key Features
- **End-to-end training**: Both RNN and classification head are trained together
- **Pre-trained Word2Vec embeddings**: Leverages Google News 300D vectors for semantic knowledge
- **Bidirectional RNN**: Captures both forward and backward context in text
- **Customizable architecture**: Full control over model design and hyperparameters
- **Comprehensive evaluation**: Detailed metrics, visualizations, and analysis

### Comparison with Experiment A

| Aspect | Experiment A | Experiment B |
|--------|-------------|--------------|
| **Model Architecture** | Pre-trained LM + MLP classifier | From-scratch RNN + classification head |
| **Embeddings** | Learned during LM pre-training | Pre-trained Word2Vec (Google News 300D) |
| **Training Approach** | Feature extraction (frozen LM) | End-to-end training |
| **Model Complexity** | Two-stage: LM + classifier | Single-stage: RNN classifier |
| **Computational Cost** | Lower (frozen backbone) | Higher (full training) |
| **Flexibility** | Limited by pre-trained LM | Fully customizable |

## Model Architecture

### RNN Classifier Components

1. **Word2Vec Embedding Layer**
   - Pre-trained Google News Word2Vec vectors (300D)
   - Automatic download and caching of model (~1.5GB)
   - Configurable freezing of embeddings
   - Handles out-of-vocabulary words with `<unk>` token

2. **RNN Layer**
   - **Type**: Configurable LSTM or GRU
   - **Architecture**: Bidirectional (default) or unidirectional
   - **Hidden dimension**: 256 (configurable)
   - **Number of layers**: 2 (configurable)
   - **Dropout**: 0.3 (configurable) between layers

3. **Classification Head**
   - **Input**: RNN output (512D for bidirectional, 256D for unidirectional)
   - **Hidden layer**: 256D with ReLU activation
   - **Output**: 2D (binary classification: negative/positive)
   - **Dropout**: 0.3 for regularization

### Key Implementation Features
- **Sequence packing**: Efficient handling of variable-length sequences
- **Gradient clipping**: Prevents exploding gradients during training
- **Early stopping**: Prevents overfitting based on validation performance
- **Learning rate scheduling**: Adaptive learning rate during training
- **Vocabulary building**: Creates vocabulary from training data with minimum frequency filtering

## File Structure

```
classification_B/
├── setup_classification_model.py    # Model setup and configuration
├── train_classification.py          # Training script
├── evaluate_classification.py       # Evaluation and analysis
├── run_experiment.sh               # Complete experiment pipeline
├── README.md                       # This file
├── word2vec_cache.pkl             # Cached Word2Vec model (~3.4GB)
└── results/                        # Output directory (created during execution)
    ├── vocab.json                  # Vocabulary mapping (49K+ tokens)
    ├── rnn_classifier_config.json  # Model configuration
    ├── trained_rnn_classifier.pth  # Trained model weights (~202MB)
    ├── training_history.png        # Training metrics visualization
    ├── confusion_matrix.png        # Confusion matrix plot
    ├── evaluation_results.json     # Detailed evaluation metrics
    └── [additional evaluation plots]
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
       --data-dir ../../data/processed_classification_data/ \
       --output-dir ./results \
       --embedding-dim 300 \
       --hidden-dim 256 \
       --num-layers 2 \
       --rnn-type lstm \
       --bidirectional \
       --min-freq 2
   ```

2. **Train Model**:
   ```bash
   python3 train_classification.py \
       --data-dir ../../data/processed_classification_data/ \
       --output-dir ./results \
       --config-file ./results/rnn_classifier_config.json \
       --epochs 10 \
       --batch-size 32 \
       --learning-rate 0.001 \
       --max-length 200
   ```

3. **Evaluate Model**:
   ```bash
   python3 evaluate_classification.py \
       --data-dir ../../data/processed_classification_data/ \
       --output-dir ./results \
       --trained-model-file trained_rnn_classifier.pth \
       --config-file ./results/rnn_classifier_config.json
   ```

## Configuration Options

### Model Parameters
- `--embedding-dim`: Word2Vec embedding dimension (default: 300)
- `--hidden-dim`: RNN hidden dimension (default: 256)
- `--num-layers`: Number of RNN layers (default: 2)
- `--dropout`: Dropout rate (default: 0.3)
- `--rnn-type`: RNN type - 'lstm' or 'gru' (default: 'lstm')
- `--bidirectional`: Use bidirectional RNN (default: True)
- `--min-freq`: Minimum word frequency for vocabulary (default: 2)

### Training Parameters
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--max-length`: Maximum sequence length (default: 200)

### Data Parameters
- `--data-dir`: Directory containing processed classification data
- `--output-dir`: Directory to save results and models
- `--device`: Device to run training on ('cpu' or 'cuda')

## Dependencies

### Core Dependencies
- **PyTorch**: Deep learning framework
- **NumPy**: Numerical computing
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical data visualization
- **Scikit-learn**: Machine learning metrics and utilities

### Specialized Dependencies
- **Gensim**: Word2Vec model loading and management
- **tqdm**: Progress bars for training loops
- **Pickle**: Model caching and serialization


## Notes

### Performance Expectations
- **Training Time**: ~30-60 minutes on CPU, ~5-15 minutes on GPU
- **Memory Usage**: ~4GB RAM (including Word2Vec cache)
- **Model Size**: ~202MB for trained model weights
- **Vocabulary Size**: ~49K tokens (with min_freq=2)

### Word2Vec Integration
- **Automatic Download**: Word2Vec model (~1.5GB) downloads automatically on first run
- **Local Caching**: Model cached in `word2vec_cache.pkl` for subsequent runs
- **Coverage**: ~85-90% of vocabulary words found in pre-trained vectors
- **Fallback**: Unknown words initialized with small random values

### Output Files
- **Model Files**: Trained weights, configuration, and vocabulary
- **Visualizations**: Training curves, confusion matrix
- **Metrics**: Detailed evaluation reports and summary statistics
- **Logs**: Training progress and performance metrics
