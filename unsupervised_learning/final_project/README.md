# AutoEncoder Final Project

A comprehensive implementation and evaluation of three autoencoder architectures for unsupervised learning tasks including denoising, latent space visualization, and anomaly detection.

## Project Overview

This project implements and compares three different autoencoder architectures:
1. **Simple AutoEncoder** - Single hidden layer architecture
2. **Multi-layer AutoEncoder** - Deep fully-connected architecture with regularization
3. **Convolutional AutoEncoder** - CNN-based architecture for image processing

## Project Structure

```
final_project/
├── encoders/                          # Autoencoder implementations
│   ├── auto_encoder_1_hidden_layer.py
│   ├── auto_encoder_many_hidden_layers.py
│   └── auto_encoder_many_conv_layers.py
├── experiments/                       # Experiment scripts
│   ├── denoising/                     # Denoising experiments
│   ├── latent_space/                  # Latent space visualization
│   └── anomaly_detection/             # Anomaly detection experiments
└── autoencoder_env/                   # Virtual environment
```

## Setup

### Prerequisites
- Python 3.7+
- Virtual environment (recommended)

### Installation
```bash
# Activate the virtual environment
source autoencoder_env/bin/activate

# Verify installation
python -c "import torch, numpy, matplotlib, sklearn; print('All packages installed')"
```

## Experiments

### 1. Denoising Experiment
Evaluates autoencoder performance on synthetic data with different noise types (Gaussian, Salt & Pepper, Speckle).

```bash
cd experiments/denoising
python autoencoder_denoising_experiment.py --epochs 50 --batch_size 64
```

**Results**: Convolutional autoencoder achieves best performance (MSE: 0.024, PSNR: 16.40 dB).

### 2. Latent Space Visualization
Visualizes the learned latent representations using t-SNE on MNIST digits.

```bash
cd experiments/latent_space
python autoencoder_latent_space_experiment.py --epochs 20
```

**Results**: Shows how different architectures learn distinct latent space representations.

### 3. Anomaly Detection
Tests anomaly detection capability using digit 9 as anomaly vs. digits 0-8 as normal data.

```bash
cd experiments/anomaly_detection
python autoencoder_anomaly_detection_experiment.py --epochs 20
```

**Results**: Modest performance (ROC AUC: 0.487-0.557) due to semantic similarity between digits.

## Key Findings

- **Convolutional AutoEncoder**: Best performance on image tasks (denoising, reconstruction)
- **Multi-layer AutoEncoder**: Good balance of capacity and regularization
- **Simple AutoEncoder**: Baseline performance, suitable for simple tasks
- **Anomaly Detection**: Challenging on similar data distributions (digits 0-8 vs. 9)

## Model Architectures

| Model | Parameters | Best Use Case |
|-------|------------|---------------|
| Simple | ~100K | Basic dimensionality reduction |
| Multi-layer | ~1.1M | Complex feature learning |
| Convolutional | ~995K | Image processing tasks |

