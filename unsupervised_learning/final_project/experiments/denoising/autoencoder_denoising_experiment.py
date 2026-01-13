#!/usr/bin/env python3
"""
AutoEncoder Denoising Experiment

This script performs a comprehensive comparison of three autoencoder architectures
for denoising tasks:

1. Simple AutoEncoder with one hidden layer
2. Multi-layer AutoEncoder with configurable hidden layers
3. Convolutional AutoEncoder with configurable conv layers

The experiment tests different noise types and levels to evaluate denoising performance.
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.manifold import TSNE
import pandas as pd
import warnings

# Suppress OpenMP compatibility warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

# Always add the project root to sys.path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from encoders.auto_encoder_1_hidden_layer import SimpleAutoEncoder, SimpleAutoEncoderTrainer, create_simple_autoencoder
from encoders.auto_encoder_many_hidden_layers import MultiLayerAutoEncoder, MultiLayerAutoEncoderTrainer, create_multi_layer_autoencoder
from encoders.auto_encoder_many_conv_layers import ConvAutoEncoder, ConvAutoEncoderTrainer, create_conv_autoencoder, ConvLayerConfig


class DenoisingExperiment:
    """Comprehensive denoising experiment for autoencoders."""
    
    def __init__(self, output_dir: str = None):
        """
        Initialize the denoising experiment.
        
        Args:
            output_dir: Directory to save experiment results. If None, saves to 'denoising_results' in the script's directory.
        """
        if output_dir is None:
            output_dir = Path(__file__).parent / "denoising_results"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Experiment configuration
        self.config = {
            'input_dim': 784,  # 28x28 flattened
            'hidden_dims': [512, 256, 128],
            'latent_dim': 64,
            'conv_input_shape': (1, 28, 28),  # 1 channel, 28x28
            'batch_size': 64,
            'learning_rate': 0.001,
            'epochs': 50,
            'noise_types': ['gaussian', 'salt_pepper', 'speckle'],
            'noise_levels': [0.1, 0.2, 0.3, 0.4, 0.5],
            'test_size': 1000
        }
        
        # Results storage
        self.results = {
            'config': self.config,
            'models': {},
            'denoising_performance': {},
            'training_metrics': {},
            'latent_analysis': {}
        }
    
    def generate_synthetic_data(self, num_samples: int = 5000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate synthetic data for denoising experiments.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Tuple of (clean_data, noisy_data)
        """
        # Generate clean synthetic patterns (simulating simple shapes)
        clean_data = torch.zeros(num_samples, self.config['input_dim'])
        
        for i in range(num_samples):
            # Create different patterns
            pattern_type = i % 4
            
            if pattern_type == 0:
                # Horizontal line
                row = (i // 28) % 28
                clean_data[i, row*28:(row+1)*28] = 1.0
            elif pattern_type == 1:
                # Vertical line
                col = (i % 28)
                clean_data[i, col::28] = 1.0
            elif pattern_type == 2:
                # Diagonal
                for j in range(28):
                    clean_data[i, j*28 + j] = 1.0
            else:
                # Random sparse pattern
                indices = torch.randperm(784)[:50]
                clean_data[i, indices] = 1.0
        
        return clean_data
    
    def add_noise(self, data: torch.Tensor, noise_type: str, noise_level: float) -> torch.Tensor:
        """
        Add noise to clean data.
        
        Args:
            data: Clean input data
            noise_type: Type of noise ('gaussian', 'salt_pepper', 'speckle')
            noise_level: Noise intensity level
            
        Returns:
            Noisy data
        """
        noisy_data = data.clone()
        
        if noise_type == 'gaussian':
            # Add Gaussian noise
            noise = torch.randn_like(data) * noise_level
            noisy_data = torch.clamp(noisy_data + noise, 0, 1)
            
        elif noise_type == 'salt_pepper':
            # Add salt and pepper noise
            mask = torch.rand_like(data) < noise_level
            salt_mask = torch.rand_like(data) < 0.5
            noisy_data[mask] = salt_mask[mask].float()
            
        elif noise_type == 'speckle':
            # Add speckle noise (multiplicative)
            noise = torch.randn_like(data) * noise_level + 1
            noisy_data = torch.clamp(noisy_data * noise, 0, 1)
        
        return noisy_data
    
    def create_models(self) -> Dict[str, Any]:
        """Create all three autoencoder models."""
        models = {}
        
        # 1. Simple AutoEncoder
        print("Creating Simple AutoEncoder...")
        simple_model = create_simple_autoencoder(
            input_dim=self.config['input_dim'],
            hidden_dim=self.config['latent_dim'],
            activation='relu'
        )
        simple_trainer = SimpleAutoEncoderTrainer(simple_model, learning_rate=self.config['learning_rate'])
        models['simple'] = {
            'model': simple_model,
            'trainer': simple_trainer,
            'name': 'Simple AutoEncoder (1 Hidden Layer)'
        }
        
        # 2. Multi-layer AutoEncoder
        print("Creating Multi-layer AutoEncoder...")
        multi_model = create_multi_layer_autoencoder(
            input_dim=self.config['input_dim'],
            hidden_dims=self.config['hidden_dims'],
            latent_dim=self.config['latent_dim'],
            activation='relu',
            dropout_rate=0.2,
            batch_norm=True
        )
        multi_trainer = MultiLayerAutoEncoderTrainer(multi_model, learning_rate=self.config['learning_rate'])
        models['multi_layer'] = {
            'model': multi_model,
            'trainer': multi_trainer,
            'name': 'Multi-layer AutoEncoder'
        }
        
        # 3. Convolutional AutoEncoder
        print("Creating Convolutional AutoEncoder...")
        conv_configs = [
            ConvLayerConfig(1, 32, kernel_size=3, pool_size=2),      # 28x28 -> 14x14
            ConvLayerConfig(32, 64, kernel_size=3, pool_size=2),     # 14x14 -> 7x7
            ConvLayerConfig(64, 128, kernel_size=3, pool_size=1),    # 7x7 -> 7x7
        ]
        
        conv_model = create_conv_autoencoder(
            input_shape=self.config['conv_input_shape'],
            conv_configs=conv_configs,
            latent_dim=self.config['latent_dim'],
            activation='relu',
            dropout_rate=0.2,
            batch_norm=True,
            use_upsampling=True
        )
        conv_trainer = ConvAutoEncoderTrainer(conv_model, learning_rate=self.config['learning_rate'])
        models['convolutional'] = {
            'model': conv_model,
            'trainer': conv_trainer,
            'name': 'Convolutional AutoEncoder'
        }
        
        return models
    
    def train_model(self, model_info: Dict[str, Any], train_data: torch.Tensor, 
                   val_data: torch.Tensor) -> Dict[str, List[float]]:
        """
        Train a single model.
        
        Args:
            model_info: Model and trainer information
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Training metrics
        """
        model = model_info['model']
        trainer = model_info['trainer']
        
        train_losses = []
        val_losses = []
        
        # Determine data format based on model type
        model_key = None
        for key, info in self.results['models'].items():
            if info['model'] == model:
                model_key = key
                break
        
        # Create data loaders with appropriate data format
        if model_key == 'convolutional':
            # Reshape data for convolutional autoencoder: [batch, 784] -> [batch, 1, 28, 28]
            train_data_reshaped = train_data.reshape(-1, 1, 28, 28)
            val_data_reshaped = val_data.reshape(-1, 1, 28, 28)
            
            train_dataset = torch.utils.data.TensorDataset(train_data_reshaped, train_data_reshaped)
            val_dataset = torch.utils.data.TensorDataset(val_data_reshaped, val_data_reshaped)
        else:
            # Use flattened data for simple and multi-layer autoencoders
            train_dataset = torch.utils.data.TensorDataset(train_data, train_data)
            val_dataset = torch.utils.data.TensorDataset(val_data, val_data)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.config['batch_size'], shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.config['batch_size'], shuffle=False
        )
        
        print(f"Training {model_info['name']}...")
        
        for epoch in range(self.config['epochs']):
            # Training
            model.train()
            epoch_train_loss = 0.0
            for batch_data, _ in train_loader:
                batch_data = batch_data.to(self.device)
                loss = trainer.train_step(batch_data)
                epoch_train_loss += loss
            
            epoch_train_loss /= len(train_loader)
            train_losses.append(epoch_train_loss)
            
            # Validation
            model.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for batch_data, _ in val_loader:
                    batch_data = batch_data.to(self.device)
                    if hasattr(trainer, 'evaluate'):
                        loss, _ = trainer.evaluate(batch_data)
                    else:
                        # For simple autoencoder
                        x_recon, _ = model(batch_data)
                        loss = nn.MSELoss()(x_recon, batch_data).item()
                    epoch_val_loss += loss
            
            epoch_val_loss /= len(val_loader)
            val_losses.append(epoch_val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config['epochs']}: "
                      f"Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses
        }
    
    def evaluate_denoising(self, model_info: Dict[str, Any], clean_data: torch.Tensor, 
                          noisy_data: torch.Tensor) -> Dict[str, float]:
        """
        Evaluate denoising performance.
        
        Args:
            model_info: Model and trainer information
            clean_data: Clean test data
            noisy_data: Noisy test data
            
        Returns:
            Denoising metrics
        """
        model = model_info['model']
        trainer = model_info['trainer']
        
        # Determine model type and format data accordingly
        model_key = None
        for key, info in self.results['models'].items():
            if info['model'] == model:
                model_key = key
                break
        
        if model_key == 'convolutional':
            # Reshape data for convolutional autoencoder
            clean_data_reshaped = clean_data.reshape(-1, 1, 28, 28)
            noisy_data_reshaped = noisy_data.reshape(-1, 1, 28, 28)
            input_data = noisy_data_reshaped.to(self.device)
        else:
            # Use flattened data for simple and multi-layer autoencoders
            input_data = noisy_data.to(self.device)
        
        model.eval()
        with torch.no_grad():
            # Get denoised output
            if hasattr(trainer, 'get_latent_representations'):
                # For multi-layer and conv autoencoders
                denoised_data = model(input_data)[0].cpu()
            else:
                # For simple autoencoder
                denoised_data = model(input_data)[0].cpu()
            
            # Reshape denoised data back to flattened format for metric calculation
            if model_key == 'convolutional':
                denoised_data = denoised_data.reshape(denoised_data.shape[0], -1)
            
            # Calculate metrics
            mse_original = mean_squared_error(clean_data.numpy().flatten(), 
                                            noisy_data.numpy().flatten())
            mse_denoised = mean_squared_error(clean_data.numpy().flatten(), 
                                            denoised_data.numpy().flatten())
            
            mae_original = mean_absolute_error(clean_data.numpy().flatten(), 
                                             noisy_data.numpy().flatten())
            mae_denoised = mean_absolute_error(clean_data.numpy().flatten(), 
                                             denoised_data.numpy().flatten())
            
            # PSNR (Peak Signal-to-Noise Ratio)
            psnr_original = 20 * np.log10(1.0 / np.sqrt(mse_original))
            psnr_denoised = 20 * np.log10(1.0 / np.sqrt(mse_denoised))
            
            # SSIM approximation (simplified)
            ssim_score = self._calculate_ssim(clean_data, denoised_data)
            
            return {
                'mse_original': mse_original,
                'mse_denoised': mse_denoised,
                'mae_original': mae_original,
                'mae_denoised': mae_denoised,
                'psnr_original': psnr_original,
                'psnr_denoised': psnr_denoised,
                'ssim': ssim_score,
                'improvement_mse': mse_original - mse_denoised,
                'improvement_mae': mae_original - mae_denoised,
                'improvement_psnr': psnr_denoised - psnr_original
            }
    
    def _calculate_ssim(self, clean_data: torch.Tensor, denoised_data: torch.Tensor) -> float:
        """Calculate simplified SSIM score."""
        # Simplified SSIM calculation
        clean_mean = clean_data.mean()
        denoised_mean = denoised_data.mean()
        
        clean_var = clean_data.var()
        denoised_var = denoised_data.var()
        
        covariance = ((clean_data - clean_mean) * (denoised_data - denoised_mean)).mean()
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * clean_mean * denoised_mean + c1) * (2 * covariance + c2)) / \
               ((clean_mean ** 2 + denoised_mean ** 2 + c1) * (clean_var + denoised_var + c2))
        
        return ssim.item()
    
    def analyze_latent_space(self, model_info: Dict[str, Any], data: torch.Tensor) -> np.ndarray:
        """Analyze latent space using t-SNE."""
        model = model_info['model']
        trainer = model_info['trainer']
        
        # Determine model type and format data accordingly
        model_key = None
        for key, info in self.results['models'].items():
            if info['model'] == model:
                model_key = key
                break
        
        if model_key == 'convolutional':
            # Reshape data for convolutional autoencoder
            data_reshaped = data.reshape(-1, 1, 28, 28)
            input_data = data_reshaped.to(self.device)
        else:
            # Use flattened data for simple and multi-layer autoencoders
            input_data = data.to(self.device)
        
        model.eval()
        with torch.no_grad():
            if hasattr(trainer, 'get_latent_representations'):
                latent_repr = trainer.get_latent_representations(input_data)
            else:
                # For simple autoencoder
                latent_repr = model.encode(input_data).cpu()
            
            # Apply t-SNE for visualization
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            latent_2d = tsne.fit_transform(latent_repr.numpy())
            
            return latent_2d
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        
        # 1. Training curves
        self._plot_training_curves()
        
        # 2. Denoising performance comparison
        self._plot_denoising_performance()
        
        # 3. Latent space visualization
        self._plot_latent_space()
        
        # 4. Sample denoising results
        self._plot_sample_results()
    
    def _plot_training_curves(self):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, (model_key, metrics) in enumerate(self.results['training_metrics'].items()):
            ax = axes[i]
            epochs = range(1, len(metrics['train_losses']) + 1)
            
            ax.plot(epochs, metrics['train_losses'], label='Training Loss', linewidth=2)
            ax.plot(epochs, metrics['val_losses'], label='Validation Loss', linewidth=2)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title(f'{self.results["models"][model_key]["name"]}\nTraining Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_denoising_performance(self):
        """Plot denoising performance across noise levels."""
        metrics = ['mse_denoised', 'psnr_denoised', 'ssim']
        metric_names = ['MSE (Lower is Better)', 'PSNR (Higher is Better)', 'SSIM (Higher is Better)']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[i]
            
            for model_key in self.results['models'].keys():
                values = []
                for noise_level in self.config['noise_levels']:
                    avg_value = np.mean([
                        self.results['denoising_performance'][model_key][noise_level][noise_type][metric]
                        for noise_type in self.config['noise_types']
                    ])
                    values.append(avg_value)
                
                ax.plot(self.config['noise_levels'], values, 
                       marker='o', linewidth=2, 
                       label=self.results['models'][model_key]['name'])
            
            ax.set_xlabel('Noise Level')
            ax.set_ylabel(metric_name)
            ax.set_title(f'Denoising Performance: {metric_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'denoising_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_latent_space(self):
        """Plot latent space visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Generate test data for latent analysis
        test_data = self.generate_synthetic_data(self.config['test_size'])
        
        for i, (model_key, model_info) in enumerate(self.results['models'].items()):
            ax = axes[i]
            
            latent_2d = self.analyze_latent_space(model_info, test_data)
            
            # Color by pattern type
            colors = ['red', 'blue', 'green', 'orange']
            pattern_types = np.arange(len(test_data)) % 4
            
            for pattern_type in range(4):
                mask = pattern_types == pattern_type
                ax.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                          c=colors[pattern_type], alpha=0.6, s=20,
                          label=f'Pattern {pattern_type}')
            
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title(f'{model_info["name"]}\nLatent Space (t-SNE)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'latent_space.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sample_results(self):
        """Plot sample denoising results."""
        # Generate test data
        test_data = self.generate_synthetic_data(16)
        noisy_data = self.add_noise(test_data, 'gaussian', 0.3)
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        
        for i in range(16):
            row = i // 4
            col = i % 4
            
            # Original
            original_img = test_data[i].reshape(28, 28).numpy()
            axes[row, col].imshow(original_img, cmap='gray')
            axes[row, col].set_title(f'Sample {i+1}')
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_originals.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Noisy and denoised for each model
        for model_key, model_info in self.results['models'].items():
            fig, axes = plt.subplots(4, 4, figsize=(16, 16))
            
            model = model_info['model']
            model.eval()
            with torch.no_grad():
                # Format data according to model type
                if model_key == 'convolutional':
                    # Reshape for convolutional autoencoder
                    noisy_data_reshaped = noisy_data.reshape(-1, 1, 28, 28)
                    denoised_data = model(noisy_data_reshaped.to(self.device))[0].cpu()
                    # Reshape back to flattened for plotting
                    denoised_data = denoised_data.reshape(denoised_data.shape[0], -1)
                else:
                    # Use flattened data for simple and multi-layer autoencoders
                    denoised_data = model(noisy_data.to(self.device))[0].cpu()
            
            for i in range(16):
                row = i // 4
                col = i % 4
                
                # Denoised
                denoised_img = denoised_data[i].reshape(28, 28).numpy()
                axes[row, col].imshow(denoised_img, cmap='gray')
                axes[row, col].set_title(f'Denoised {i+1}')
                axes[row, col].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.output_dir / f'sample_denoised_{model_key}.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def generate_report(self) -> str:
        """Generate comprehensive experiment report."""
        report = []
        report.append("=" * 80)
        report.append("AUTOENCODER DENOISING EXPERIMENT REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Configuration
        report.append("EXPERIMENT CONFIGURATION:")
        report.append("-" * 40)
        for key, value in self.config.items():
            report.append(f"{key}: {value}")
        report.append("")
        
        # Model architectures
        report.append("MODEL ARCHITECTURES:")
        report.append("-" * 40)
        for model_key, model_info in self.results['models'].items():
            model = model_info['model']
            report.append(f"{model_info['name']}:")
            report.append(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            report.append("")
        
        # Denoising performance summary
        report.append("DENOISING PERFORMANCE SUMMARY:")
        report.append("-" * 40)
        
        for model_key, model_info in self.results['models'].items():
            report.append(f"\n{model_info['name']}:")
            
            # Average performance across all noise types and levels
            all_mse = []
            all_psnr = []
            all_ssim = []
            
            for noise_level in self.config['noise_levels']:
                for noise_type in self.config['noise_types']:
                    metrics = self.results['denoising_performance'][model_key][noise_level][noise_type]
                    all_mse.append(metrics['mse_denoised'])
                    all_psnr.append(metrics['psnr_denoised'])
                    all_ssim.append(metrics['ssim'])
            
            report.append(f"  Average MSE: {np.mean(all_mse):.6f}")
            report.append(f"  Average PSNR: {np.mean(all_psnr):.2f} dB")
            report.append(f"  Average SSIM: {np.mean(all_ssim):.4f}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_results(self):
        """Save all experiment results."""
        # Save configuration
        with open(self.output_dir / 'experiment_config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save denoising performance
        with open(self.output_dir / 'denoising_performance.json', 'w') as f:
            json.dump(self.results['denoising_performance'], f, indent=2)
        
        # Save training metrics
        with open(self.output_dir / 'training_metrics.json', 'w') as f:
            json.dump(self.results['training_metrics'], f, indent=2)
        
        # Save report
        report = self.generate_report()
        with open(self.output_dir / 'experiment_report.txt', 'w') as f:
            f.write(report)
        
        print(f"Results saved to: {self.output_dir}")
    
    def run_experiment(self):
        """Run the complete denoising experiment."""
        print("Starting AutoEncoder Denoising Experiment...")
        print("=" * 60)
        
        # Generate data
        print("Generating synthetic data...")
        train_data = self.generate_synthetic_data(3000)
        val_data = self.generate_synthetic_data(1000)
        test_data = self.generate_synthetic_data(self.config['test_size'])
        
        # Create models
        print("Creating autoencoder models...")
        self.results['models'] = self.create_models()
        
        # Train models
        print("Training models...")
        for model_key, model_info in self.results['models'].items():
            print(f"\nTraining {model_info['name']}...")
            metrics = self.train_model(model_info, train_data, val_data)
            self.results['training_metrics'][model_key] = metrics
        
        # Evaluate denoising performance
        print("\nEvaluating denoising performance...")
        self.results['denoising_performance'] = {}
        
        for model_key, model_info in self.results['models'].items():
            print(f"Evaluating {model_info['name']}...")
            self.results['denoising_performance'][model_key] = {}
            
            for noise_level in self.config['noise_levels']:
                self.results['denoising_performance'][model_key][noise_level] = {}
                
                for noise_type in self.config['noise_types']:
                    print(f"  Testing {noise_type} noise at level {noise_level}...")
                    noisy_test_data = self.add_noise(test_data, noise_type, noise_level)
                    
                    metrics = self.evaluate_denoising(model_info, test_data, noisy_test_data)
                    self.results['denoising_performance'][model_key][noise_level][noise_type] = metrics
        
        # Create visualizations
        print("\nCreating visualizations...")
        self.create_visualizations()
        
        # Save results
        print("\nSaving results...")
        self.save_results()
        
        # Print summary
        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(self.generate_report())


def main():
    """Main function to run the experiment."""
    parser = argparse.ArgumentParser(description='AutoEncoder Denoising Experiment')
    parser.add_argument('--output_dir', type=str, default='./denoising_results',
                       help='Directory to save experiment results')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    
    args = parser.parse_args()
    
    # Create and run experiment
    experiment = DenoisingExperiment(output_dir=args.output_dir)
    experiment.config['epochs'] = args.epochs
    experiment.config['batch_size'] = args.batch_size
    
    experiment.run_experiment()


if __name__ == "__main__":
    main() 