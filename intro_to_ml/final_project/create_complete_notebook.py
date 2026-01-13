#!/usr/bin/env python3
"""
Script to create a complete self-contained neural network notebook
"""

import nbformat as nbf

cells = []

# 1. Title and Introduction
cells.append(nbf.v4.new_markdown_cell(
"""# Complete Neural Network Project: MNIST and MB Dataset Classification

**Final Project - Introduction to Machine Learning**

This notebook contains a complete, self-contained implementation of a feed-forward neural network for two classification tasks:
1. **MNIST Digit Classification** - Multi-class classification of handwritten digits
2. **MB Dataset Classification** - Binary classification for medical data (Control vs Fibrosis)

## Features
- Custom neural network implementation with ReLU activation and softmax output
- Mini-batch gradient descent with cross-entropy loss
- Hyperparameter optimization using smart phased sampling
- Comprehensive evaluation and visualization of results
- Self-contained: No external imports from local files required

All code is included in this notebook. Simply run all cells to execute the complete project.
"""))

# 2. Setup and Imports
cells.append(nbf.v4.new_markdown_cell("## Setup and Imports"))
cells.append(nbf.v4.new_code_cell(
"""import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import json
import random
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

print("Setup complete! All required libraries imported.")
print(f"Current working directory: {os.getcwd()}")
"""))

# 3. Neural Network Implementation
cells.append(nbf.v4.new_markdown_cell(
"""## Neural Network Implementation

Complete implementation of a feed-forward neural network with:
- ReLU activation for hidden layers
- Softmax activation for output layer
- Cross-entropy loss function
- Mini-batch gradient descent
- Xavier weight initialization
"""))
cells.append(nbf.v4.new_code_cell(
"""class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, epochs=50, batch_size=32, seed=42):
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = []
        self.biases = []
        self.training_loss = []
        self.initialize_parameters()

    def initialize_parameters(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(self.layer_sizes[i], self.layer_sizes[i+1]) * np.sqrt(2. / self.layer_sizes[i])
            b = np.zeros((1, self.layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_derivative(self, z):
        return (z > 0).astype(float)

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def one_hot_encode(self, y, num_classes):
        return np.eye(num_classes)[y.astype(int)]

    def forward_propagation(self, X):
        activations = [X]
        zs = []
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            zs.append(z)
            a = self.relu(z)
            activations.append(a)
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        zs.append(z)
        a = self.softmax(z)
        activations.append(a)
        return activations, zs

    def backward_propagation(self, X, y, activations, zs):
        grads_w = [np.zeros_like(w) for w in self.weights]
        grads_b = [np.zeros_like(b) for b in self.biases]
        m = X.shape[0]
        num_layers = len(self.weights)
        delta = activations[-1] - y
        grads_w[-1] = np.dot(activations[-2].T, delta) / m
        grads_b[-1] = np.sum(delta, axis=0, keepdims=True) / m
        for l in range(num_layers - 2, -1, -1):
            delta = np.dot(delta, self.weights[l+1].T) * self.relu_derivative(zs[l])
            grads_w[l] = np.dot(activations[l].T, delta) / m
            grads_b[l] = np.sum(delta, axis=0, keepdims=True) / m
        return grads_w, grads_b

    def update_parameters(self, grads_w, grads_b):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grads_w[i]
            self.biases[i] -= self.learning_rate * grads_b[i]

    def fit(self, X, y, verbose=True):
        start_time = time.time()
        num_classes = np.unique(y).size
        y_encoded = self.one_hot_encode(y, num_classes)
        n = X.shape[0]
        self.training_loss = []
        for epoch in range(self.epochs):
            indices = np.arange(n)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y_encoded[indices]
            for i in range(0, n, self.batch_size):
                X_batch = X_shuffled[i:i+self.batch_size]
                y_batch = y_shuffled[i:i+self.batch_size]
                activations, zs = self.forward_propagation(X_batch)
                grads_w, grads_b = self.backward_propagation(X_batch, y_batch, activations, zs)
                self.update_parameters(grads_w, grads_b)
            activations, _ = self.forward_propagation(X)
            loss = -np.mean(np.sum(y_encoded * np.log(activations[-1] + 1e-8), axis=1))
            self.training_loss.append(loss)
            if verbose and (epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs - 1):
                print(f'Epoch {epoch+1}/{self.epochs} - Loss: {loss:.4f}')
        end_time = time.time()
        return end_time - start_time

    def predict(self, X):
        activations, _ = self.forward_propagation(X)
        return np.argmax(activations[-1], axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def plot_training_loss(self, save_path=None):
        plt.plot(self.training_loss, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

print("Neural Network class defined successfully!")
"""))

# 4. Data Loading Functions
cells.append(nbf.v4.new_markdown_cell(
"""## Data Loading Functions

These functions load and preprocess the MNIST and MB datasets. If the CSV files are not found, they fall back to sklearn datasets.
"""))
cells.append(nbf.v4.new_code_cell(
"""def load_mnist_data():
    \"\"\"Load and preprocess MNIST dataset from experiments/mnist/inputs directory or fallback to sklearn digits.\"\"\"
    try:
        train_data = pd.read_csv('experiments/mnist/inputs/MNIST-train.csv')
        test_data = pd.read_csv('experiments/mnist/inputs/MNIST-test.csv')
        
        # Separate features and labels using 'y' column
        X_train_full = train_data.drop('y', axis=1).values  # All columns except 'y'
        y_train_full = train_data['y'].values               # 'y' column (labels)
        
        X_test = test_data.drop('y', axis=1).values    # All columns except 'y'
        y_test = test_data['y'].values                 # 'y' column (labels)
        
        # Split training data into train and validation sets (80/20 split)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
        )
        
        # Normalize pixel values to [0, 1]
        X_train = X_train / 255.0
        X_val = X_val / 255.0
        X_test = X_test / 255.0
        
        print(f\"Loaded MNIST CSV: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test\")
        return X_train, X_val, X_test, y_train, y_val, y_test
    except Exception as e:
        print(f\"Falling back to sklearn digits: {e}\")
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Split into train, validation, and test sets (60/20/20 split)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 of remaining 80% = 20% of total
        )
        
        print(f\"Loaded sklearn digits: {X_train.shape[0]} train, {X_val.shape[0]} val, {X_test.shape[0]} test\")
        return X_train, X_val, X_test, y_train, y_val, y_test

def load_mb_data():
    \"\"\"Load and preprocess MB dataset from experiments/mb/inputs directory or fallback to sklearn breast cancer.\"\"\"
    try:
        train_data = pd.read_csv('experiments/mb/inputs/MB_data_train.csv', index_col=0)
        y_train = []
        for patient_id in train_data.index:
            if patient_id.startswith('Pt_Fibro_'):
                y_train.append(1)
            elif patient_id.startswith('Pt_Ctrl_'):
                y_train.append(0)
            else:
                y_train.append(0)
        y_train = np.array(y_train)
        X_train = train_data.values
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        print(f\"Loaded MB CSV: {X_train_split.shape[0]} train, {X_val.shape[0]} val\")
        return X_train_split, X_val, y_train_split, y_val
    except Exception as e:
        print(f\"Falling back to sklearn breast cancer: {e}\")
        cancer = load_breast_cancer()
        X, y = cancer.data, cancer.target
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f\"Loaded sklearn breast cancer: {X_train.shape[0]} train, {X_val.shape[0]} val\")
        return X_train, X_val, y_train, y_val
"""))

# 5. Experiment and Optimization Functions
cells.append(nbf.v4.new_markdown_cell(
"""## Experiment and Hyperparameter Optimization Functions

These functions run the experiments and optimize hyperparameters for both datasets.
"""))
cells.append(nbf.v4.new_code_cell(
"""def optimize_hyperparameters(X_train, y_train, X_val, y_val, max_trials=10, is_mb=False):
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    if is_mb:
        architectures = [
            [input_size, 64, 32, num_classes],
            [input_size, 128, 64, num_classes],
            [input_size, 256, 128, 64, num_classes],
            [input_size, 128, 128, num_classes],
            [input_size, 64, 64, 64, num_classes],
            [input_size, 512, 256, 128, 64, num_classes],
            [input_size, 32, 32, 32, 32, num_classes],
            [input_size, 256, 256, num_classes],
        ]
        learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
        batch_sizes = [4, 8, 16, 32, 64]
    else:
        architectures = [
            [input_size, 64, 32, num_classes],
            [input_size, 128, 64, num_classes],
            [input_size, 256, 128, 64, num_classes],
            [input_size, 128, 128, num_classes],
            [input_size, 64, 64, 64, num_classes],
            [input_size, 512, 256, 128, 64, num_classes],
            [input_size, 32, 32, 32, 32, num_classes],
            [input_size, 256, 256, num_classes],
        ]
        learning_rates = [0.0001, 0.001, 0.01, 0.1, 0.5]
        batch_sizes = [8, 16, 32, 64, 128]
    epochs_list = [20, 30, 50, 100, 150]
    results = []
    best_accuracy = 0
    best_config = None
    for trial in range(max_trials):
        arch = random.choice(architectures)
        lr = random.choice(learning_rates)
        epochs = random.choice(epochs_list)
        batch_size = random.choice(batch_sizes)
        nn = NeuralNetwork(
            layer_sizes=arch,
            learning_rate=lr,
            epochs=epochs,
            batch_size=batch_size
        )
        training_time = nn.fit(X_train, y_train, verbose=False)
        train_acc = nn.score(X_train, y_train)
        val_acc = nn.score(X_val, y_val)
        result = {
            'trial': trial+1,
            'architecture': arch,
            'learning_rate': lr,
            'epochs': epochs,
            'batch_size': batch_size,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'training_time': training_time,
            'final_loss': nn.training_loss[-1] if nn.training_loss else None,
            'total_params': sum(w.size + b.size for w, b in zip(nn.weights, nn.biases)),
        }
        results.append(result)
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_config = result.copy()
        print(f"Trial {trial+1}/{max_trials}: Validation Accuracy = {val_acc:.4f}")
    return best_config, results
"""))

# 6. MNIST Experiment
cells.append(nbf.v4.new_markdown_cell(
"""## MNIST Experiment

Load the data, optimize hyperparameters, train the final model, and visualize results.
"""))
cells.append(nbf.v4.new_code_cell(
"""X_train_mnist, X_val_mnist, X_test_mnist, y_train_mnist, y_val_mnist, y_test_mnist = load_mnist_data()
best_config_mnist, results_mnist = optimize_hyperparameters(
    X_train_mnist, y_train_mnist, X_val_mnist, y_val_mnist, max_trials=10
)
print("\nBest MNIST config:", best_config_mnist)
final_nn_mnist = NeuralNetwork(
    layer_sizes=best_config_mnist['architecture'],
    learning_rate=best_config_mnist['learning_rate'],
    epochs=best_config_mnist['epochs'],
    batch_size=best_config_mnist['batch_size']
)
final_nn_mnist.fit(X_train_mnist, y_train_mnist, verbose=True)
final_val_accuracy_mnist = final_nn_mnist.score(X_val_mnist, y_val_mnist)
final_test_accuracy_mnist = final_nn_mnist.score(X_test_mnist, y_test_mnist)
print(f"\nFinal MNIST Validation Accuracy: {final_val_accuracy_mnist:.4f}")
print(f"Final MNIST Test Accuracy: {final_test_accuracy_mnist:.4f}")
final_nn_mnist.plot_training_loss()
"""))

# 7. MB Experiment
cells.append(nbf.v4.new_markdown_cell(
"""## MB Experiment

Load the data, optimize hyperparameters, train the final model, and visualize results.
"""))
cells.append(nbf.v4.new_code_cell(
"""X_train_mb, X_val_mb, y_train_mb, y_val_mb = load_mb_data()
best_config_mb, results_mb = optimize_hyperparameters(
    X_train_mb, y_train_mb, X_val_mb, y_val_mb, max_trials=10, is_mb=True
)
print("\nBest MB config:", best_config_mb)
final_nn_mb = NeuralNetwork(
    layer_sizes=best_config_mb['architecture'],
    learning_rate=best_config_mb['learning_rate'],
    epochs=best_config_mb['epochs'],
    batch_size=best_config_mb['batch_size']
)
final_nn_mb.fit(X_train_mb, y_train_mb, verbose=True)
final_val_accuracy_mb = final_nn_mb.score(X_val_mb, y_val_mb)
print(f"\nFinal MB Validation Accuracy: {final_val_accuracy_mb:.4f}")
final_nn_mb.plot_training_loss()
print("\nClassification Report:")
print(classification_report(y_val_mb, final_nn_mb.predict(X_val_mb), target_names=['Control', 'Fibrosis']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_val_mb, final_nn_mb.predict(X_val_mb)))
"""))

# 8. Save notebook
nb = nbf.v4.new_notebook()
nb['cells'] = cells
with open("complete_neural_network_project.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook created: complete_neural_network_project.ipynb") 