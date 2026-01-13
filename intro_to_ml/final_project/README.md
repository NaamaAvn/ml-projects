# Neural Network Implementation with Experiments

This project implements a fully connected feed-forward neural network from scratch and provides separate experiments for different datasets.

## Project Structure

```
final_project/
├── neural_network.py              # Core NeuralNetwork class
├── experiments/
│   ├── mnist/                     # MNIST experiment
│   │   ├── inputs/                # MNIST data files
│   │   ├── outputs/
│   │   │   ├── plots/             # Training and optimization plots
│   │   │   └── optimization_results/ # Optimization results JSON files
│   │   └── mnist_experiment.py    # MNIST experiment script
│   └── mb/                        # MB dataset experiment
│       ├── inputs/                # MB data files
│       ├── outputs/
│       │   ├── plots/             # Training and optimization plots
│       │   └── optimization_results/ # Optimization results JSON files
│       └── mb_experiment.py       # MB experiment script
├── inputs/                        # Original data files (for reference)
├── outputs/                       # Original outputs (for reference)
├── test_mb.py                     # Quick test script for MB dataset
└── final_project.py               # Original combined script (for reference)
```

## Core Neural Network Features

The `NeuralNetwork` class in `neural_network.py` provides:

- **Architecture**: Fully connected feed-forward neural network
- **Activation Functions**: ReLU for hidden layers, Softmax for output layer
- **Loss Function**: Cross-entropy loss
- **Optimization**: Mini-batch gradient descent
- **Initialization**: He initialization for better gradient flow
- **Methods**: `fit()`, `predict()`, `score()`, `plot_training_loss()`

## Experiments

### MNIST Experiment (`experiments/mnist/`)

**Dataset**: MNIST handwritten digits (or sklearn digits as fallback)
- **Task**: Multi-class classification (10 classes)
- **Data Split**: Train/Test (80/20)
- **Features**: 784 pixel values (normalized to [0,1])

**Usage**:
```bash
cd experiments/mnist

# Run training with default or best configuration
python mnist_experiment.py

# Run hyperparameter optimization
python mnist_experiment.py optimize
```

**Outputs**:
- Training loss plots in `outputs/plots/`
- Optimization results and plots in `outputs/optimization_results/`
- Best configuration automatically loaded for subsequent runs

### MB Dataset Experiment (`experiments/mb/`)

**Dataset**: MB biological dataset (or sklearn breast cancer as fallback)
- **Task**: Binary classification (Control vs Fibrosis)
- **Data Split**: Train/Validation (80/20) - since only one file available
- **Features**: 1620 gene expression features (standardized)

**Usage**:
```bash
cd experiments/mb

# Run training with default or best configuration
python mb_experiment.py

# Run hyperparameter optimization
python mb_experiment.py optimize
```

**Outputs**:
- Training loss plots in `outputs/plots/`
- Optimization results and plots in `outputs/optimization_results/`
- Classification reports and confusion matrices
- Best configuration automatically loaded for subsequent runs

## Hyperparameter Optimization

Both experiments use a smart phased optimization strategy:

1. **Phase 1**: Test different architectures with default parameters
2. **Phase 2**: Test learning rates with the best architecture
3. **Phase 3**: Test epochs with the best architecture and learning rate
4. **Phase 4**: Test batch sizes with the best configuration
5. **Phase 5**: Random combinations to fill remaining trials

**Optimization Parameters**:
- **Architectures**: Various depths and widths (8 different configurations)
- **Learning Rates**: 5 different values (adapted per dataset)
- **Epochs**: 5 different values (20-150)
- **Batch Sizes**: 5 different values (adapted per dataset)
- **Total Trials**: 25 combinations

## Data Requirements

### MNIST Experiment
Place your MNIST data files in `experiments/mnist/inputs/`:
- `MNIST-train.csv` - Training data with 'y' column for labels
- `MNIST-test.csv` - Test data with 'y' column for labels

If files are not found, the experiment will fall back to sklearn digits dataset.

### MB Experiment
Place your MB data file in `experiments/mb/inputs/`:
- `MB_data_train.csv` - Training data with patient IDs as row names

The script will automatically:
- Extract labels from row names (`Pt_Fibro_*` = Fibrosis, `Pt_Ctrl_*` = Control)
- Split into train/validation sets (80/20)
- Apply standard scaling to features

If the file is not found, the experiment will fall back to sklearn breast cancer dataset.

## Quick Test

To quickly test the MB dataset functionality:

```bash
python test_mb.py
```

This will run a simple training session with the MB dataset and show results.

## Key Features

1. **Modular Design**: Separate experiments for different datasets
2. **Automatic Fallbacks**: Uses sklearn datasets if original data not available
3. **Smart Optimization**: Phased hyperparameter search strategy
4. **Comprehensive Outputs**: Plots, results, and metrics saved automatically
5. **Configuration Persistence**: Best configurations saved and reloaded
6. **Binary Classification Support**: Special handling for MB dataset with classification reports
7. **Validation Split**: Proper train/validation split for MB dataset

## Requirements

- Python 3.7+
- numpy
- matplotlib
- scikit-learn
- pandas (for CSV loading)

## Notes

- The MB dataset uses train/validation split since only one data file is available
- The MNIST dataset uses train/test split as it has separate files
- Both experiments automatically create output directories as needed
- Optimization results are timestamped and saved as JSON files
- The neural network implementation is from scratch (no external deep learning libraries) 