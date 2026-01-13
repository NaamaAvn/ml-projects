#!/usr/bin/env python3
"""
AutoEncoder Anomaly Detection Experiment (MNIST)

This experiment trains all three autoencoder architectures (simple, multi-layer, convolutional)
on MNIST digits 0-8, and tests anomaly detection on digit 9 using reconstruction error.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from encoders.auto_encoder_1_hidden_layer import create_simple_autoencoder, SimpleAutoEncoderTrainer
from encoders.auto_encoder_many_hidden_layers import create_multi_layer_autoencoder, MultiLayerAutoEncoderTrainer
from encoders.auto_encoder_many_conv_layers import create_conv_autoencoder, ConvAutoEncoderTrainer, ConvLayerConfig

def load_mnist_0_8_vs_9(n_train=10000, n_test=2000):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist['data'] / 255.0
    y = mnist['target'].astype(int)
    # Normal: 0-8, Anomaly: 9
    X_normal = X[y != 9]
    y_normal = y[y != 9]
    X_anomaly = X[y == 9]
    y_anomaly = y[y == 9]
    # Split normal into train/test
    X_train, X_normal_test, y_train, y_normal_test = train_test_split(X_normal, y_normal, train_size=n_train, random_state=42, stratify=y_normal)
    # Take a subset of anomalies for test
    X_anomaly_test = X_anomaly[:n_test]
    y_anomaly_test = y_anomaly[:n_test]
    # Test set: normal + anomaly
    X_test = np.vstack([X_normal_test, X_anomaly_test])
    y_test = np.concatenate([np.zeros_like(y_normal_test), np.ones_like(y_anomaly_test)])  # 0=normal, 1=anomaly
    return X_train, X_test, y_test

def train_autoencoder(model, trainer, X, epochs=20, batch_size=128, device="cpu"):
    model.to(device)
    trainer.model = model
    dataset = torch.utils.data.TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(X, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch, _ in loader:
            batch = batch.to(device)
            loss = trainer.train_step(batch)
            total_loss += loss
        avg_loss = total_loss / len(loader)
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.5f}")
    return model

def get_reconstruction_errors(model, X, device="cpu"):
    model.eval()
    errors = []
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        batch_size = 256
        for i in range(0, len(X), batch_size):
            batch = X_tensor[i:i+batch_size]
            if hasattr(model, "encode"):
                recon, _ = model(batch)
            else:
                recon, _ = model(batch)
            # Handle both 1D (flattened) and 4D (image) data
            if batch.dim() == 4:  # Image data [batch, channels, height, width]
                err = ((batch - recon) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            else:  # Flattened data [batch, features]
                err = ((batch - recon) ** 2).mean(dim=1).cpu().numpy()
            errors.append(err)
    return np.concatenate(errors)

def plot_error_hist(errors, y_test, model_name, output_dir):
    plt.figure(figsize=(8, 5))
    plt.hist(errors[y_test == 0], bins=50, alpha=0.6, label='Normal (0-8)')
    plt.hist(errors[y_test == 1], bins=50, alpha=0.6, label='Anomaly (9)')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Count')
    plt.title(f'Reconstruction Error Histogram ({model_name})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f'error_hist_{model_name}.png')
    plt.close()

def plot_roc_curve(y_true, errors, model_name, output_dir):
    fpr, tpr, _ = roc_curve(y_true, errors)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({model_name})')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_dir / f'roc_{model_name}.png')
    plt.close()
    return roc_auc

def main():
    parser = argparse.ArgumentParser(description="AutoEncoder Anomaly Detection Experiment (MNIST)")
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--n_train', type=int, default=10000, help='Number of normal training samples')
    parser.add_argument('--n_test', type=int, default=2000, help='Number of anomaly test samples')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "anomaly_detection_results"
    output_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    X_train, X_test, y_test = load_mnist_0_8_vs_9(n_train=args.n_train, n_test=args.n_test)
    X_train_img = X_train.reshape(-1, 1, 28, 28)
    X_test_img = X_test.reshape(-1, 1, 28, 28)

    # Model configs
    model_types = [
        ("simple", create_simple_autoencoder, SimpleAutoEncoderTrainer, {"hidden_dim": 32, "activation": "relu"}),
        ("multi_layer", create_multi_layer_autoencoder, MultiLayerAutoEncoderTrainer, {"hidden_dims": [128, 64], "latent_dim": 32, "activation": "relu", "dropout_rate": 0.2, "batch_norm": True}),
        ("convolutional", create_conv_autoencoder, ConvAutoEncoderTrainer, {"latent_dim": 32, "activation": "relu", "dropout_rate": 0.2, "batch_norm": True, "use_upsampling": True}),
    ]

    results = {}
    for model_name, model_fn, trainer_cls, kwargs in model_types:
        print(f"\nTraining {model_name} autoencoder...")
        if model_name == "convolutional":
            conv_configs = [
                ConvLayerConfig(1, 32, kernel_size=3, pool_size=2),
                ConvLayerConfig(32, 64, kernel_size=3, pool_size=2),
                ConvLayerConfig(64, 128, kernel_size=3, pool_size=1),
            ]
            model = model_fn(
                input_shape=(1, 28, 28),
                conv_configs=conv_configs,
                **kwargs
            )
            X_tr = X_train_img
            X_te = X_test_img
        elif model_name == "simple":
            model = model_fn(
                input_dim=784,
                hidden_dim=kwargs["hidden_dim"],
                activation=kwargs["activation"]
            )
            X_tr = X_train
            X_te = X_test
        else:  # multi_layer
            model = model_fn(
                input_dim=784,
                hidden_dims=kwargs["hidden_dims"],
                latent_dim=kwargs["latent_dim"],
                activation=kwargs["activation"],
                dropout_rate=kwargs["dropout_rate"],
                batch_norm=kwargs["batch_norm"]
            )
            X_tr = X_train
            X_te = X_test
        trainer = trainer_cls(model, learning_rate=0.001)
        model = train_autoencoder(model, trainer, X_tr, epochs=args.epochs, batch_size=args.batch_size, device=device)
        errors = get_reconstruction_errors(model, X_te, device=device)
        roc_auc = plot_roc_curve(y_test, errors, model_name, output_dir)
        plot_error_hist(errors, y_test, model_name, output_dir)
        print(f"{model_name} ROC AUC: {roc_auc:.3f}")
        results[model_name] = roc_auc

    # Save summary
    with open(output_dir / "summary.txt", "w") as f:
        for model_name, auc_score in results.items():
            f.write(f"{model_name}: ROC AUC = {auc_score:.3f}\n")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 