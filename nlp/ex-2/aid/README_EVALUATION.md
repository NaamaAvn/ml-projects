# Language Model Evaluation

This directory contains a comprehensive evaluation script for the trained language model with multiple evaluation modes and metrics.

## Overview

The evaluation script (`evaluate.py`) provides a complete assessment of the language model's performance using industry-standard metrics and multiple evaluation modes:

### **Evaluation Modes:**

1. **Full Mode** (Default): Comprehensive evaluation with all metrics
2. **Test Mode**: Basic testing (like the old test.py script)
3. **Generate Mode**: Text generation only

### **Metrics Available:**

- **Perplexity**: Standard language model evaluation metric
- **BLEU Score**: Measures text generation quality (BLEU-1, BLEU-2, BLEU-3, BLEU-4)
- **ROUGE Score**: Measures text generation quality (ROUGE-1, ROUGE-2, ROUGE-L)
- **Token-level Accuracy**: Measures prediction accuracy at the token level

## Features

### 📊 Multiple Evaluation Modes

1. **Full Mode** (`--mode full`): Complete evaluation with all metrics
   - Perplexity calculation
   - Token-level accuracy
   - BLEU scores (1-4 gram)
   - ROUGE scores (precision, recall, F1)
   - Sample text generation and comparison

2. **Test Mode** (`--mode test`): Basic testing (replaces test.py)
   - Loss and accuracy calculation
   - Sample text generation
   - Quick evaluation for development

3. **Generate Mode** (`--mode generate`): Text generation only
   - Generate sample texts with various prompts
   - No evaluation metrics
   - Useful for quick text generation testing

### 🎯 Text Generation Evaluation

The evaluator generates text samples using various prompts and compares them against reference samples from the test set to calculate BLEU and ROUGE scores.

### 📈 Comprehensive Reporting

- Detailed metrics with formatted output
- Sample generated texts for qualitative assessment
- Results saved to JSON file for further analysis
- Different output formats for different modes

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

The evaluation script requires these additional dependencies:
- `nltk`: For BLEU score calculation and text tokenization
- `rouge-score`: For ROUGE score calculation

### 2. Download NLTK Data

The script will automatically download required NLTK data on first run:
- `punkt` tokenizer
- `wordnet` corpus (for METEOR score)

## Usage

### Quick Start

```bash
# Run comprehensive evaluation (default)
python evaluate.py

# Run basic test mode (like old test.py)
python evaluate.py --mode test

# Run text generation only
python evaluate.py --mode generate
```

### Advanced Usage

```bash
# Comprehensive evaluation with custom parameters
python evaluate.py \
    --mode full \
    --model-path ./model/language_model.pth \
    --vocab-path ./data/vocab.json \
    --data-dir ./data/ \
    --output-path ./evaluation_results.json \
    --batch-size 32 \
    --num-samples 100 \
    --device cpu

# Basic test mode with custom parameters
python evaluate.py \
    --mode test \
    --num-samples 10 \
    --batch-size 16

# Text generation with more samples
python evaluate.py \
    --mode generate \
    --num-samples 20
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model-path` | `./model/language_model.pth` | Path to trained model |
| `--vocab-path` | `./data/vocab.json` | Path to vocabulary file |
| `--data-dir` | `./data/` | Directory containing test data |
| `--output-path` | `./evaluation_results.json` | Path to save evaluation results |
| `--batch-size` | `32` | Batch size for evaluation |
| `--num-samples` | `100` | Number of samples for text generation evaluation |
| `--device` | `cpu` | Device to run evaluation on (`cpu`/`cuda`) |
| `--mode` | `full` | Evaluation mode: `full`, `test`, or `generate` |

### Testing the Evaluation

Run the test script to verify everything works:

```bash
python test_evaluation.py
```

This will run a quick test with a small subset of data to ensure all components work correctly.

## Output

### Console Output by Mode

#### Full Mode Output:
```
🔬 COMPREHENSIVE EVALUATION MODE
==================================================
Starting comprehensive model evaluation...

1. Calculating Perplexity...
2. Calculating Token-level Accuracy...
3. Generating Text Samples...
4. Extracting Reference Samples...
5. Calculating BLEU Scores...
6. Calculating ROUGE Scores...

============================================================
LANGUAGE MODEL EVALUATION RESULTS
============================================================

📊 BASIC METRICS:
   Perplexity: 1977.0226
   Token-level Accuracy: 0.1195 (11.95%)

🎯 BLEU SCORES:
   BLEU-1: 0.1957
   BLEU-2: 0.0508
   BLEU-3: 0.0167
   BLEU-4: 0.0092

🔍 ROUGE SCORES:
   ROUGE-1 Precision: 0.2317
   ROUGE-1 Recall: 0.1816
   ROUGE-1 F1: 0.1962
   ...
```

#### Test Mode Output:
```
🧪 BASIC TEST MODE
==================================================
Creating test data loader...
Running basic evaluation...

📊 BASIC TEST RESULTS:
   Test Loss: 7.5801
   Test Accuracy: 0.1195 (11.95%)

📝 SAMPLE TEXT GENERATION:
Generating 5 sample texts:
============================================================
Sample 1:
Prompt: the movie was
Generated: the movie was just great potential...
```

#### Generate Mode Output:
```
🎭 TEXT GENERATION MODE
==================================================
Generating 3 sample texts:
============================================================
Sample 1:
Prompt: the movie was
Generated: the movie was just great potential...
```

### JSON Results File

Results are saved to a JSON file containing all metrics and sample texts:

```json
{
  "perplexity": 1977.0226,
  "accuracy": 0.1195,
  "bleu_scores": {
    "bleu_1": 0.1957,
    "bleu_2": 0.0508,
    "bleu_3": 0.0167,
    "bleu_4": 0.0092
  },
  "rouge_scores": {
    "rouge1_precision": 0.2317,
    "rouge1_recall": 0.1816,
    "rouge1_fmeasure": 0.1962,
    ...
  },
  "generated_samples": [...],
  "reference_samples": [...]
}
```

## Understanding the Metrics

### Perplexity
- **Range**: 1 to ∞ (lower is better)
- **Interpretation**: Measures how surprised the model is by the test data
- **Good values**: 50-200 for language models
- **Formula**: exp(cross_entropy_loss)

### BLEU Score
- **Range**: 0 to 1 (higher is better)
- **Interpretation**: Measures n-gram overlap between generated and reference text
- **Good values**: 0.1-0.3 for language models
- **Note**: BLEU-4 is most commonly used

### ROUGE Score
- **Range**: 0 to 1 (higher is better)
- **Interpretation**: Measures recall and precision of n-grams
- **Good values**: 0.1-0.3 for language models
- **Components**: Precision, Recall, F1-measure

### Token-level Accuracy
- **Range**: 0 to 1 (higher is better)
- **Interpretation**: Percentage of correctly predicted next tokens
- **Good values**: 15-30% for language models
- **Note**: This is typically low due to the large vocabulary size

## Performance Considerations

### Memory Usage
- Large batch sizes may cause memory issues
- Reduce `--batch-size` if you encounter memory errors
- Use CPU if GPU memory is insufficient

### Speed
- Evaluation can take several minutes depending on data size
- Use smaller `--num-samples` for faster testing
- GPU acceleration available with `--device cuda`
- Test mode is faster than full mode

### Data Requirements
- Test data must be in the expected format
- Vocabulary must match the trained model
- Model file must be compatible with the current code

## Migration from test.py

If you were previously using `test.py`, you can now use the equivalent functionality in `evaluate.py`:

### Old test.py usage:
```bash
python test.py --num-samples 5
```

### New evaluate.py usage:
```bash
python evaluate.py --mode test --num-samples 5
```

### Benefits of the new approach:
- **No code duplication**: All functionality in one script
- **More comprehensive**: Full evaluation available when needed
- **Better organization**: Clear separation of modes
- **Consistent interface**: Same command-line arguments across modes

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install nltk rouge-score
   ```

2. **NLTK Data Missing**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('wordnet')
   ```

3. **Memory Errors**
   - Reduce batch size: `--batch-size 16`
   - Use CPU: `--device cpu`

4. **File Not Found**
   - Check file paths are correct
   - Ensure model and vocabulary files exist

### Error Messages

- `Model file not found`: Check `--model-path` argument
- `Vocabulary file not found`: Check `--vocab-path` argument
- `Test data file not found`: Check `--data-dir` argument
- `CUDA not available`: Use `--device cpu`

## Integration with Training Pipeline

The evaluation script is designed to work seamlessly with the training pipeline:

1. **After Training**: Run evaluation to assess model performance
2. **Model Comparison**: Compare different models using the same metrics
3. **Hyperparameter Tuning**: Use metrics to guide model improvements
4. **Research**: Use results for academic or research purposes

## Customization

### Adding New Metrics

To add new evaluation metrics, extend the `LanguageModelEvaluator` class:

```python
def calculate_custom_metric(self, test_loader):
    # Your custom metric implementation
    pass
```

### Modifying Text Generation

Customize text generation by modifying the `generate_text_samples` method:

```python
def generate_text_samples(self, num_samples=100, max_length=50, temperature=1.0):
    # Custom generation logic
    pass
```

## Contributing

When adding new features to the evaluation script:

1. Add comprehensive docstrings
2. Include error handling
3. Add tests to `test_evaluation.py`
4. Update this README with new features
5. Ensure backward compatibility

## License

This evaluation script is part of the language model project and follows the same license terms. 