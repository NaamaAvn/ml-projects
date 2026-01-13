# Language Model Implementation

This directory contains a complete implementation of a Recurrent Neural Network (LSTM) based language model for next word prediction on the IMDB dataset.

## Overview

The language model is designed to:
- Learn sequential dependencies in text data
- Predict the next word given a sequence of previous words
- Generate coherent text samples
- Evaluate model performance using perplexity and accuracy metrics

## Text Classification Experiments

This project also includes two different approaches for text classification on the IMDB dataset:

- **Experiment A**: Pre-trained Language Model + Classification Head (Feature Extraction)
- **Experiment B**: From-scratch RNN Classifier with Word2Vec Embeddings (End-to-End Training)

📊 **See [EXPERIMENT_COMPARISON.md](EXPERIMENT_COMPARISON.md) for detailed comparison and results analysis.**

## Architecture

The model consists of the following components:

1. **Embedding Layer**: Converts token indices to dense vector representations
2. **LSTM Layers**: Captures sequential dependencies in the input data
3. **Dropout Layers**: Provides regularization to prevent overfitting
4. **Output Layer**: Predicts probability distribution over the vocabulary

### Model Parameters

- **Embedding Dimension**: 128 (configurable)
- **Hidden Dimension**: 256 (configurable)
- **Number of LSTM Layers**: 2 (configurable)
- **Dropout Rate**: 0.3 (configurable)
- **Vocabulary Size**: Dynamic (based on training data)

## Files

### Main Implementation
- `process_imdb_lm.py`: Complete implementation including data processing, model definition, and training
- `demo_language_model.py`: Demo script for testing and using the trained model

### Key Classes

#### LanguageModel
The main model class with the following methods:

- `__init__()`: Initialize the model with specified parameters
- `forward()`: Forward pass through the model
- `generate_text()`: Generate text samples from prompts
- `train_model()`: Complete training pipeline with validation
- `train_epoch()`: Train for one epoch
- `validate_epoch()`: Validate for one epoch
- `calculate_loss()`: Calculate cross-entropy loss
- `plot_training_history()`: Visualize training progress

## Usage

### Quick Training Options

For faster training and testing, use the quick training script:

```bash
# Interactive menu
python quick_train.py

# Or specify mode directly
python quick_train.py ultra    # Ultra-fast demo (5% data, 3 epochs)
python quick_train.py fast     # Fast training (10% data, 5 epochs)
python quick_train.py medium   # Medium training (25% data, 10 epochs)
python quick_train.py full     # Full training (100% data, 15 epochs)
```

### Training Modes

1. **Ultra-fast Demo** (5% data, 3 epochs) - ~2-3 minutes
   - Perfect for testing the setup
   - Uses minimal data and epochs

2. **Fast Training** (10% data, 5 epochs) - ~5-10 minutes
   - Good for development and testing
   - Reasonable performance in short time

3. **Medium Training** (25% data, 10 epochs) - ~15-25 minutes
   - Better performance while still fast
   - Good balance of speed and quality

4. **Full Training** (100% data, 15 epochs) - ~1-2 hours
   - Production-quality model
   - Best performance but takes longer

### 1. Training the Model (Advanced Options)

```bash
# Fast training with custom parameters
python process_imdb_lm.py --mode fast

# Full training with all data
python process_imdb_lm.py --mode full

# Custom training with specific parameters
python process_imdb_lm.py --mode custom --data-ratio 0.2 --epochs 8 --fast-params
```

### 2. Running the Demo

```bash
python demo_language_model.py
```

This will:
- Load the trained model
- Demonstrate text generation with different prompts
- Analyze model performance
- Provide interactive text generation

### 3. Using the Model Programmatically

```python
from process_imdb_lm import LanguageModel, load_and_test_model

# Load a trained model
model = load_and_test_model('./models/language_model.pth')

# Generate text
generated_text = model.generate_text(
    start_tokens=['the', 'movie', 'was'],
    max_length=30,
    temperature=0.8
)
print(generated_text)
```

## Performance Optimization

### Fast Training Parameters

When using fast training mode, the model uses these optimized parameters:

- **Embedding Dimension**: 64 (vs 128 in full mode)
- **Hidden Dimension**: 128 (vs 256 in full mode)
- **LSTM Layers**: 1 (vs 2 in full mode)
- **Dropout**: 0.2 (vs 0.3 in full mode)
- **Learning Rate**: 0.01 (vs 0.001 in full mode)
- **Batch Size**: 64 (vs 32 in full mode)

### Data Subset Training

You can train on a subset of the data for faster iteration:

```python
# Train on 10% of data and test on 10% of test data
main(data_subset_ratio=0.1, test_subset_ratio=0.1, max_epochs=5)

# Train on 25% of data and test on 25% of test data
main(data_subset_ratio=0.25, test_subset_ratio=0.25, max_epochs=10)

# Train on 50% of data but test on only 10% of test data
main(data_subset_ratio=0.5, test_subset_ratio=0.1, max_epochs=10)
```

### Command Line Options

```bash
# Custom training with specific data ratios
python process_imdb_lm.py --mode custom --data-ratio 0.2 --test-ratio 0.1 --epochs 8

# Fast training with minimal test data
python process_imdb_lm.py --mode custom --data-ratio 0.1 --test-ratio 0.05 --epochs 5
```

### Intermediate Data Saving

The implementation automatically saves intermediate outputs from steps 1 and 2, so you can skip data preprocessing on subsequent runs:

- **First run**: Full preprocessing (takes longer)
- **Subsequent runs**: Load existing processed data (much faster)

Files saved:
- `data_splits.json`: Train/val/test data splits
- `tokenizer_info.json`: Tokenizer configuration
- `vocab.json`: Vocabulary mapping
- `dataloader_configs.json`: DataLoader configurations
- `train.json`, `val.json`, `test.json`: Processed sequences

## Features

### Text Generation
- **Temperature Sampling**: Control randomness in text generation
- **Custom Prompts**: Start generation with any sequence of words
- **Length Control**: Specify maximum generation length
- **Stop Conditions**: Automatically stop at end tokens

### Training Features
- **Progress Tracking**: Real-time training progress with tqdm
- **Model Checkpointing**: Save best model based on validation loss
- **Gradient Clipping**: Prevent exploding gradients
- **Early Stopping**: Save best model during training
- **Training History**: Plot loss and accuracy curves

### Evaluation Metrics
- **Perplexity**: Standard language model evaluation metric
- **Accuracy**: Token-level prediction accuracy
- **Loss Tracking**: Training and validation loss monitoring

## Model Configuration

You can customize the model by modifying these parameters in `process_imdb_lm.py`:

```python
model = LanguageModel(
    vocab=vocab,
    embedding_dim=128,    # Word embedding dimension
    hidden_dim=256,       # LSTM hidden state dimension
    num_layers=2,         # Number of LSTM layers
    dropout=0.3,          # Dropout rate
    device=device         # CPU or CUDA device
)
```

## Data Processing

The implementation includes comprehensive data processing:

1. **Text Cleaning**: Remove extra spaces, normalize punctuation
2. **Tokenization**: Basic English tokenizer
3. **Vocabulary Building**: Minimum frequency filtering
4. **Sequence Creation**: Input-target pairs for language modeling
5. **Data Analysis**: Visualizations and statistics

## Output Files

After training, the following files will be created:

- `./models/language_model.pth`: Trained model checkpoint
- `./models/training_history.png`: Training curves plot
- `./data/processed_lm_data/`: Processed dataset files
- `./data/processed_lm_data/plots/`: Dataset analysis plots

## Performance

Typical performance metrics on the IMDB dataset:
- **Training Time**: ~30-60 minutes (depending on hardware)
- **Validation Loss**: ~4.5-5.5 (after 15 epochs)
- **Perplexity**: ~90-150
- **Accuracy**: ~15-25% (token-level)

## Requirements

- PyTorch >= 1.9.0
- torchtext >= 0.10.0
- matplotlib
- seaborn
- tqdm
- numpy

## Notes

- The model uses the IMDB dataset for movie review text
- Special tokens (`<unk>`, `<pad>`, `<sos>`, `<eos>`) are included in vocabulary
- The implementation supports both CPU and GPU training
- Gradient clipping is applied to prevent training instability
- The model automatically handles padding tokens in loss calculation

## Troubleshooting

1. **Out of Memory**: Reduce batch size or model dimensions
2. **Slow Training**: Use GPU if available, reduce sequence length
3. **Poor Generation**: Increase training epochs, adjust temperature
4. **Model Not Loading**: Ensure the model file exists and is compatible

## Future Improvements

- Add attention mechanisms
- Implement beam search for text generation
- Add support for different RNN architectures (GRU, Transformer)
- Implement subword tokenization
- Add model compression techniques 