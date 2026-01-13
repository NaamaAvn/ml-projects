#!/usr/bin/env python3
"""
AutoEncoder Latent Space Experiment (MNIST)

This experiment trains all three autoencoder architectures (simple, multi-layer, convolutional)
on the MNIST dataset, with both 2D and 3D latent spaces. It visualizes the learned latent
representations directly (for 2D/3D) and with t-SNE (for higher dimensions), using
the t-SNE implementation from exercise_2_Q4_tsne.py.
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
from sklearn.datasets import load_digits

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from encoders.auto_encoder_1_hidden_layer import create_simple_autoencoder, SimpleAutoEncoderTrainer
from encoders.auto_encoder_many_hidden_layers import create_multi_layer_autoencoder, MultiLayerAutoEncoderTrainer
from encoders.auto_encoder_many_conv_layers import create_conv_autoencoder, ConvAutoEncoderTrainer, ConvLayerConfig

# --- t-SNE implementation (adapted from exercise_2_Q4_tsne.py) ---
from sklearn.metrics import pairwise_distances

def compute_high_dim_similarities(X, perplexity=30.0):
    (n, d) = X.shape
    D = pairwise_distances(X, squared=True)
    P = np.zeros((n, n))
    logU = np.log(perplexity)
    for i in range(n):
        beta = 1.0
        betamin = -np.inf
        betamax = np.inf
        Di = np.delete(D[i], i)
        H, thisP = compute_entropy(Di, beta)
        H_diff = H - logU
        tries = 0
        while np.abs(H_diff) > 1e-5 and tries < 50:
            if H_diff > 0:
                betamin = beta
                beta = 2.0 * beta if betamax == np.inf else (beta + betamax) / 2.
            else:
                betamax = beta
                beta = beta / 2.0 if betamin == -np.inf else (beta + betamin) / 2.
            H, thisP = compute_entropy(Di, beta)
            H_diff = H - logU
            tries += 1
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP
    return P

def compute_entropy(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    if sumP < 1e-20:
        return 0.0, np.ones_like(P) / len(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    return H, P / sumP

def compute_low_dim_similarities_tsne(Y):
    D = pairwise_distances(Y, squared=True)
    Q = 1.0 / (1.0 + D)
    np.fill_diagonal(Q, 0.0)
    sum_Q = np.sum(Q)
    if sum_Q < 1e-20:
        sum_Q = 1e-20
    return Q / sum_Q

def tsne(X, dim=2, perplexity=30.0, learning_rate=200.0, max_iter=1000):
    n, d = X.shape
    Y = np.random.normal(0, 1e-4, (n, dim))
    P = compute_high_dim_similarities(X, perplexity)
    P = (P + P.T) / (2.0 * n)
    P = np.maximum(P, 1e-12)
    dY_prev = np.zeros_like(Y)
    for iter in range(max_iter):
        Q = compute_low_dim_similarities_tsne(Y)
        Q = np.maximum(Q, 1e-12)
        dY = np.zeros_like(Y)
        for i in range(n):
            diff = Y[i] - Y
            squared_distances = np.sum(diff ** 2, axis=1)
            pq_diff = P[i] - Q[i]
            weights = 1.0 / (1.0 + squared_distances)
            dY[i] = 4.0 * np.sum((pq_diff * weights).reshape(-1, 1) * diff, axis=0)
        grad_norm = np.linalg.norm(dY)
        if grad_norm > 100:
            dY = dY / grad_norm * 100
        momentum = 0.5 if iter < 250 else 0.8
        dY = momentum * dY_prev + learning_rate * dY
        Y = Y - dY
        dY_prev = dY
        Y = Y - np.mean(Y, axis=0)
        if iter % 100 == 0:
            kl_div = np.sum(P * np.log(np.clip(P / Q, 1e-12, 1e12)))
            print(f"Iteration {iter}, KL divergence: {kl_div:.4f}")
    return Y

# --- Experiment code ---

def load_digits_data(flatten=True, n_samples=2000):
    digits = load_digits()
    X = digits.data[:n_samples]
    y = digits.target[:n_samples]
    if not flatten:
        X = X.reshape(-1, 1, 8, 8)
    return X, y

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

def get_latent_representations(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        if hasattr(model, "encode"):
            latent = model.encode(X_tensor)
        else:
            _, latent = model(X_tensor)
        return latent.cpu().numpy()

def plot_latent_2d(latent, labels, title, save_path):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent[:, 0], latent[:, 1], c=labels, cmap="tab10", s=10, alpha=0.7)
    plt.colorbar(scatter, ticks=range(10))
    plt.title(title)
    plt.xlabel("Latent 1")
    plt.ylabel("Latent 2")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_latent_3d(latent, labels, title, save_path):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2], c=labels, cmap="tab10", s=10, alpha=0.7)
    fig.colorbar(scatter, ticks=range(10))
    ax.set_title(title)
    ax.set_xlabel("Latent 1")
    ax.set_ylabel("Latent 2")
    ax.set_zlabel("Latent 3")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="AutoEncoder Latent Space Experiment (Digits)")
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save results')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--n_samples', type=int, default=1000, help='Number of digits samples to use')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "latent_space_results"
    output_dir.mkdir(exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Digits
    X_flat, y = load_digits_data(flatten=True, n_samples=args.n_samples)
    X_img, _ = load_digits_data(flatten=False, n_samples=args.n_samples)

    # Model configs
    latent_dims = [2, 3]
    model_types = [
        ("simple", create_simple_autoencoder, SimpleAutoEncoderTrainer, {"activation": "relu"}),
        ("multi_layer", create_multi_layer_autoencoder, MultiLayerAutoEncoderTrainer, {"activation": "relu", "dropout_rate": 0.2, "batch_norm": True}),
        ("convolutional", create_conv_autoencoder, ConvAutoEncoderTrainer, {"activation": "relu", "dropout_rate": 0.2, "batch_norm": True, "use_upsampling": True}),
    ]

    for latent_dim in latent_dims:
        for model_name, model_fn, trainer_cls, extra_kwargs in model_types:
            print(f"\nTraining {model_name} autoencoder with latent_dim={latent_dim}...")
            if model_name == "convolutional":
                conv_configs = [
                    ConvLayerConfig(1, 16, kernel_size=3, pool_size=2),
                    ConvLayerConfig(16, 32, kernel_size=3, pool_size=1),
                ]
                model = model_fn(
                    input_shape=(1, 8, 8),
                    conv_configs=conv_configs,
                    latent_dim=latent_dim,
                    **extra_kwargs
                )
                X = X_img
            elif model_name == "simple":
                model = model_fn(
                    input_dim=64,
                    hidden_dim=latent_dim,
                    **extra_kwargs
                )
                X = X_flat
            else:  # multi_layer
                model = model_fn(
                    input_dim=64,
                    hidden_dims=[32, 16],
                    latent_dim=latent_dim,
                    **extra_kwargs
                )
                X = X_flat
            trainer = trainer_cls(model, learning_rate=0.001)
            model = train_autoencoder(model, trainer, X, epochs=args.epochs, batch_size=args.batch_size, device=device)
            latent = get_latent_representations(model, X, device=device)
            # Plot
            if latent_dim == 2:
                plot_latent_2d(latent, y, f"{model_name} AE latent space (2D)", output_dir / f"latent_{model_name}_2d.png")
            elif latent_dim == 3:
                plot_latent_3d(latent, y, f"{model_name} AE latent space (3D)", output_dir / f"latent_{model_name}_3d.png")

    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 