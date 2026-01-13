# Language Modeling Pipeline on IMDB Dataset

This project implements a modular language modeling pipeline using LSTM architecture on the IMDB movie reviews dataset. The pipeline is organized into building blocks that can be run independently or as part of the complete workflow.

## Project Structure

```
nlp-course-project/
├── main.py                           # Main orchestration script
├── data/
│   ├── 01_load_and_split_data.py     # Step 1: Load and split IMDB data
│   ├── 02_analyze_data.py            # Step 2: EDA and data analysis
│   ├── 03_create_vocabulary.py       # Step 3: Build vocabulary
│   ├── 04_lm_create_sequences.py      # Step 4: Create LM sequences and dataloaders
│   └── processed_lm_data/            # Processed data directory
│       ├── data_splits.json          # Train/val/test splits
│       ├── vocab.json                # Vocabulary mapping
│       ├── train.json                # Training sequences
│       ├── val.json                  # Validation sequences
│       ├── test.json                 # Test sequences
│       ├── dataset_stats.json        # Dataset statistics
│       └── eda_plots/                # EDA visualizations
├── model/
│   ├── 05_train_language_model.py    # Step 5: Train the language model
│   ├── 06_test_model.py              # Step 6: Test the trained model
│   ├── language_model.pth            # Trained model checkpoint
│   ├── training_history.png          # Training curves
│   └── test_results.json             # Test results
└── README_LM_PIPELINE.md             # This file
```

## Pipeline Steps

### Step 1: Load and Split Data (`data/01_load_and_split_data.py`)
- Loads IMDB dataset from torchtext
- Splits training data into train (90%) and validation (10%) sets
- Saves data splits to JSON files

### Step 2: Analyze Data (`data/02_analyze_data.py`)
- Performs exploratory data analysis
- Creates visualizations:
  - Sequence length distribution
  - Label distribution (positive/negative reviews)
  - Word frequency analysis
  - Word length distribution
- Saves analysis statistics

### Step 3: Create Vocabulary (`data/03_create_vocabulary.py`)
- Builds vocabulary from training data
- Includes special tokens: `<unk>`, `<pad>`, `<sos>`, `<eos>`
- Sets minimum frequency threshold (default: 2)
- Saves vocabulary mapping

### Step 4: Create Sequences (`data/04_lm_create_sequences.py`)
- Creates input-target pairs for language modeling
- Processes text into sequences of specified length
- Creates PyTorch DataLoaders for train/val/test sets
- Saves processed sequences to JSON files

### Step 5: Train Language Model (`model/05_train_language_model.py`)
- Implements LSTM-based language model
- Trains model with configurable parameters
- Saves best model based on validation loss
- Creates training history plots

### Step 6: Test Model (`model/06_test_model.py`)
- Evaluates trained model on test set
- Generates sample texts using the model
- Saves test results and metrics

## Usage

### Running the Complete Pipeline

```bash
# Run full pipeline
python main.py --full

# Run full pipeline in fast mode (smaller model, faster training)
python main.py --full --fast

# Run with custom parameters
python main.py --full --epochs 15 --batch-size 64
```

### Running Individual Steps

```bash
# Run specific step
python main.py --step 3

# Run step with custom parameters
python main.py --step 5 --epochs 20 --batch-size 32 --fast
```

### Checking Pipeline Status

```bash
# Check which steps have been completed
python main.py --status
```

### Running Steps Directly

You can also run individual scripts directly:

```bash
# Step 1: Load and split data
python data/01_load_and_split_data.py

# Step 2: Analyze data
python data/02_analyze_data.py

# Step 3: Create vocabulary
python data/03_create_vocabulary.py

# Step 4: Create sequences
python data/04_lm_create_sequences.py --batch-size 32

# Step 5: Train model
python model/05_train_language_model.py --epochs 10 --batch-size 32

# Step 6: Test model
python model/06_test_model.py --batch-size 32
```

## Model Architecture

The language model uses an LSTM-based architecture:

- **Embedding Layer**: Converts token indices to dense vectors
- **LSTM Layers**: Captures sequential dependencies (configurable layers)
- **Dropout**: Regularization to prevent overfitting
- **Output Layer**: Linear layer to predict next word probabilities

### Default Parameters

- Embedding dimension: 128
- Hidden dimension: 256
- Number of LSTM layers: 2
- Dropout rate: 0.3
- Learning rate: 0.001
- Batch size: 32
- Sequence length: 50

### Fast Mode Parameters

- Embedding dimension: 64
- Hidden dimension: 128
- Number of LSTM layers: 1
- Dropout rate: 0.2
- Learning rate: 0.01

## Output Files

### Data Processing Outputs
- `data_splits.json`: Raw data splits
- `vocab.json`: Vocabulary mapping
- `train.json`, `val.json`, `test.json`: Processed sequences
- `dataset_stats.json`: Dataset statistics
- `eda_plots/`: EDA visualizations

### Model Outputs
- `language_model.pth`: Trained model checkpoint
- `training_history.png`: Training/validation curves
- `test_results.json`: Test set performance metrics

## Dependencies

The pipeline requires the following Python packages:
- torch
- torchtext
- matplotlib
- seaborn
- numpy
- tqdm
- argparse

Install dependencies using:
```bash
pip install torch torchtext matplotlib seaborn numpy tqdm
```

## Features

### Modular Design
- Each step is independent and can be run separately
- Clear input/output dependencies between steps
- Easy to modify or extend individual components

### Flexible Configuration
- Configurable model parameters
- Fast mode for quick experimentation
- Customizable data processing parameters

### Comprehensive Logging
- Detailed progress tracking
- Clear error messages
- Pipeline status checking

### Reproducible Results
- Fixed random seeds
- Deterministic data splitting
- Consistent preprocessing

## Example Workflow

1. **Quick Start** (Fast Mode):
   ```bash
   python main.py --full --fast
   ```

2. **Production Training**:
   ```bash
   python main.py --full --epochs 20 --batch-size 64
   ```

3. **Incremental Development**:
   ```bash
   # Run data processing
   python main.py --step 1
   python main.py --step 2
   python main.py --step 3
   python main.py --step 4
   
   # Experiment with different model parameters
   python main.py --step 5 --epochs 5 --fast
   python main.py --step 6
   ```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: Ensure all required packages are installed
2. **Out of Memory**: Reduce batch size or use fast mode
3. **Step Dependencies**: Check that previous steps have completed successfully
4. **File Permissions**: Ensure write permissions for output directories

### Debugging

- Use `--status` to check pipeline progress
- Run individual steps to isolate issues
- Check log messages for detailed error information

## Extending the Pipeline

### Adding New Steps
1. Create a new script in the appropriate directory
2. Follow the naming convention: `XX_step_name.py`
3. Add step configuration to `main.py`
4. Update prerequisites checking

### Modifying Model Architecture
1. Edit the `LanguageModel` class in training script
2. Update model configuration parameters
3. Ensure compatibility with saved checkpoints

### Adding New Analysis
1. Extend the analysis functions in `02_analyze_data.py`
2. Add new visualizations or statistics
3. Update the main analysis workflow

## Performance Considerations

- **Fast Mode**: Use for development and testing
- **GPU Training**: Automatically detected and used if available
- **Memory Management**: Adjust batch size based on available memory
- **Data Subsetting**: Can be implemented for faster iteration

## License

This project is part of an NLP course assignment and follows the course guidelines for academic use. 