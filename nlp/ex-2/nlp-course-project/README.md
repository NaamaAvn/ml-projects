# NLP Course Final Project

## Overview

This project is a comprehensive pipeline for training and evaluating a Language Model on the IMDB movie review dataset. The pipeline handles everything from initial data loading and processing to model training, evaluation, and results visualization. It is designed to be modular, allowing each step of the process to be run independently or as part of a complete workflow.

## Functionality

The project is broken down into several key steps, each handled by a dedicated script:

1.  **Data Loading and Splitting**: Loads the raw IMDB dataset and splits it into training, validation, and test sets.
2.  **Exploratory Data Analysis (EDA)**: Analyzes the dataset to generate statistics (e.g., review length, word frequency) and creates visualizations to understand the data distribution.
3.  **Vocabulary Creation**: Builds a vocabulary from the training data, mapping each unique token to an index.
4.  **Sequence Creation**: Converts the text reviews into sequences of token indices suitable for training a language model.
5.  **Model Training**: Trains an LSTM-based language model to predict the next word in a sequence. It saves the best model, its configuration, and training history plots.
6.  **Model Evaluation**: Evaluates the trained model on the test set using a variety of metrics, including:
    *   Perplexity
    *   BLEU Score
    *   ROUGE Score
    *   Token-level Accuracy
    *   Test Loss

## Installation | Requirements

To get started, install the required dependencies.

1.  **Create and activate a virtual environment (recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file includes all necessary libraries such as `torch`, `matplotlib`, `nltk`, `rouge-score`, `tqdm`, and `aiofiles`.

3.  **Download NLTK data:**
    The evaluation script will automatically download the necessary NLTK data (`punkt` and `wordnet`) on its first run.

## Usage / Quickstart

The entire pipeline is orchestrated through the `main.py` script. You can run the full pipeline or individual steps.

### Running the Full Pipeline

To run the entire process from data loading to evaluation in one command:

```bash
python main.py --full
```

For a quicker run with a smaller model, use the `--fast` flag:

```bash
python main.py --full --fast
```

### Running Individual Steps

You can also run each step of the pipeline individually. This is useful for debugging or if you only need to re-run a specific part of the process.

```bash
# Step 1: Load and Split Data
python main.py --step 1

# Step 2: Analyze Data
python main.py --step 2

# Step 3: Create Vocabulary
python main.py --step 3

# Step 4: Create Sequences for Language Model
python main.py --step 4

# Step 5: Train the Model
python main.py --step 5

# Step 6: Evaluate the Model
python main.py --step 6
```

### Checking Pipeline Status

To see which steps have been completed, you can check the status:
```bash
python main.py --status
```
This will show which output files have been generated for each step.

### Command-Line Arguments

The `main.py` script accepts several arguments to customize the pipeline's execution.

#### Main Arguments

| Argument | Description |
|---|---|
| `--full` | Runs the complete pipeline from Step 1 to Step 6. |
| `--step <1-6>` | Runs a single, specific step of the pipeline. |
| `--status`| Checks and prints the completion status of each pipeline step. |

#### General Configuration

These arguments control the file paths for inputs and outputs.

| Argument | Default | Description |
|---|---|---|
| `--data-dir` | `./data/` | Directory for all data-related files. |
| `--model-dir` | `./model/` | Directory to save the trained model. |
| `--plots-dir` | `./plots/` | Directory to save output plots. |
| `--results-dir`| `./results/`| Directory to save evaluation results. |

#### Training and Data Arguments

These arguments are used to configure the training process and specify the amount of data to use. They are primarily used in Step 5 (Training).

| Argument | Default | Description |
|---|---|---|
| `--fast`| `False` | If specified, uses a smaller model configuration for faster training. Ideal for testing and debugging. |
| `--epochs`| `10`| The number of complete passes through the training dataset. |
| `--batch-size`| `32`| The number of training examples utilized in one iteration. |
| `--data-ratio`| `1.0` | The proportion of the training and validation data to use (e.g., `0.5` for 50%). |
| `--test-ratio`| `1.0` | The proportion of the test data to use. |
| `--use-async`| `False` | Use async training for better latency (experimental feature). |

## Directory Structure

The project is organized into the following directories to maintain a clean and scalable structure:

```
nlp-course-project/
├── data/              # Contains raw, processed, and split data files.
├── model/             # Stores the trained model weights and configuration.
├── plots/             # Contains output plots, like training history graphs.
├── results/           # Stores evaluation results and metrics in JSON format.
├── scripts/           # Holds all data processing and helper scripts.
├── experiments/       # Contains classification experiments (A & B).
│   ├── classification_A/     # Pre-trained LM as feature extractor.
│   ├── classification_B/     # From-scratch RNN with Word2Vec.
│   ├── comparison_results/   # Results from comparing classification experiments.
│   └── experiments_comparison.py  # Script to compare experiment results.
├── comparison_results/ # Results from comparing different approaches.
├── main.py            # The main entry point to run the pipeline.
├── train.py           # Contains the LanguageModel class and training logic.
├── evaluate.py        # Contains the model evaluation logic.
├── config.py          # Centralized configuration file.
├── requirements.txt   # Lists all project dependencies.
└── README.md          # You are here!
```

## Additional Features

### Classification Experiments

The project includes two additional classification experiments that build upon the trained language model:

- **Classification Experiment A**: Uses the pre-trained language model as a feature extractor with a frozen backbone
- **Classification Experiment B**: Implements a from-scratch RNN classifier with pre-trained Word2Vec embeddings

Both experiments can be found in the `experiments/` directory with their own README files.

### Configuration Management

The project uses a centralized configuration system (`config.py`) that allows easy modification of:
- Model hyperparameters
- Training settings
- Data processing parameters
- File paths and directories

### Fast Mode

The `--fast` flag enables a smaller model configuration for quick testing:
- Reduced embedding dimension (64 vs 128)
- Smaller hidden dimension (128 vs 256)
- Fewer layers (1 vs 2)
- Higher learning rate (0.01 vs 0.001)
- Fewer training epochs (5 vs 10)

### Async Training

The `--use-async` flag enables experimental async training for better latency during training.

## Model Architecture

The language model uses an LSTM-based architecture with:
- **Embedding Layer**: Converts token indices to dense vectors
- **LSTM Layers**: Processes sequential data with configurable layers
- **Dropout**: Regularization to prevent overfitting
- **Output Layer**: Linear layer for next-token prediction

## Performance

The model achieves competitive performance on the IMDB dataset with:
- **Perplexity**: Measures how well the model predicts the next word
- **BLEU Score**: Evaluates text generation quality
- **ROUGE Score**: Measures text summarization quality
- **Accuracy**: Token-level prediction accuracy

