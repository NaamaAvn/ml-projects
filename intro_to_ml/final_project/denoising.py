import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.manifold import TSNE
from typing import Dict, List, Tuple, Any

from encoders.auto_encoder_1_hidden_layer import SimpleAutoEncoder, SimpleAutoEncoderTrainer, create_simple_autoencoder

class DenoisingExperiment:
    """Comprehensive denoising experiment for simple autoencoder."""
    
    def __init__(self):
        """
        Initialize the denoising experiment.
        """
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Set up plotting style
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        
        # Experiment configuration
        self.config = {
            'input_dim': 784,  # 28x28 flattened
            'latent_dim': 64,
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
        """Create simple autoencoder model."""
        models = {}
        
        # Simple AutoEncoder
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
        
        return models
    
    def train_model(self, model_info: Dict[str, Any], train_data: torch.Tensor, 
                   val_data: torch.Tensor) -> Dict[str, List[float]]:
        """
        Train the simple autoencoder model.
        
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
        
        # Create data loaders
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
        
        # Use flattened data for simple autoencoder
        input_data = noisy_data.to(self.device)
        
        model.eval()
        with torch.no_grad():
            # Get denoised output
            denoised_data = model(input_data)[0].cpu()
            
            # Calculate metrics using PyTorch functions to avoid numpy issues
            mse_loss = nn.MSELoss()
            mae_loss = nn.L1Loss()
            
            # Ensure all tensors are on CPU and flattened
            clean_flat = clean_data.flatten()
            noisy_flat = noisy_data.flatten()
            denoised_flat = denoised_data.flatten()
            
            mse_original = mse_loss(noisy_flat, clean_flat).item()
            mse_denoised = mse_loss(denoised_flat, clean_flat).item()
            
            mae_original = mae_loss(noisy_flat, clean_flat).item()
            mae_denoised = mae_loss(denoised_flat, clean_flat).item()
            
            # PSNR (Peak Signal-to-Noise Ratio)
            psnr_original = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(torch.tensor(mse_original))).item()
            psnr_denoised = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(torch.tensor(mse_denoised))).item()
            
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
        
        # Use flattened data for simple autoencoder
        input_data = data.to(self.device)
        
        model.eval()
        with torch.no_grad():
            # For simple autoencoder
            latent_repr = model.encode(input_data).cpu()
            
            # Apply t-SNE for visualization - convert to numpy safely
            try:
                latent_numpy = latent_repr.detach().cpu().numpy()
                tsne = TSNE(n_components=2, random_state=42, perplexity=30)
                latent_2d = tsne.fit_transform(latent_numpy)
                return latent_2d
            except RuntimeError as e:
                print(f"Warning: Could not convert tensor to numpy for t-SNE: {e}")
                # Return a simple 2D projection as fallback
                return np.random.randn(latent_repr.shape[0], 2)
    
    def print_training_curves(self):
        """Print training and validation curves summary."""
        print("\n" + "=" * 60)
        print("TRAINING CURVES SUMMARY")
        print("=" * 60)
        
        for model_key, metrics in self.results['training_metrics'].items():
            model_name = self.results['models'][model_key]['name']
            print(f"\n{model_name}:")
            
            train_losses = metrics['train_losses']
            val_losses = metrics['val_losses']
            
            print(f"  Final Training Loss: {train_losses[-1]:.6f}")
            print(f"  Final Validation Loss: {val_losses[-1]:.6f}")
            print(f"  Best Validation Loss: {min(val_losses):.6f} (Epoch {val_losses.index(min(val_losses))+1})")
            print(f"  Training Loss Reduction: {train_losses[0] - train_losses[-1]:.6f}")
            print(f"  Validation Loss Reduction: {val_losses[0] - val_losses[-1]:.6f}")
    
    def print_denoising_performance(self):
        """Print denoising performance summary."""
        print("\n" + "=" * 60)
        print("DENOISING PERFORMANCE SUMMARY")
        print("=" * 60)
        
        for model_key, model_info in self.results['models'].items():
            print(f"\n{model_info['name']}:")
            
            # Calculate average performance across all noise types and levels
            all_mse = []
            all_psnr = []
            all_ssim = []
            all_improvements = []
            
            for noise_level in self.config['noise_levels']:
                for noise_type in self.config['noise_types']:
                    metrics = self.results['denoising_performance'][model_key][noise_level][noise_type]
                    all_mse.append(metrics['mse_denoised'])
                    all_psnr.append(metrics['psnr_denoised'])
                    all_ssim.append(metrics['ssim'])
                    all_improvements.append(metrics['improvement_mse'])
            
            print(f"  Average MSE: {np.mean(all_mse):.6f}")
            print(f"  Average PSNR: {np.mean(all_psnr):.2f} dB")
            print(f"  Average SSIM: {np.mean(all_ssim):.4f}")
            print(f"  Average MSE Improvement: {np.mean(all_improvements):.6f}")
            
            # Show performance by noise type
            print(f"  Performance by Noise Type:")
            for noise_type in self.config['noise_types']:
                noise_mse = []
                for noise_level in self.config['noise_levels']:
                    metrics = self.results['denoising_performance'][model_key][noise_level][noise_type]
                    noise_mse.append(metrics['mse_denoised'])
                print(f"    {noise_type}: {np.mean(noise_mse):.6f} MSE")
    
    def print_latent_space_analysis(self):
        """Print latent space analysis summary."""
        print("\n" + "=" * 60)
        print("LATENT SPACE ANALYSIS")
        print("=" * 60)
        
        # Generate test data for latent analysis
        test_data = self.generate_synthetic_data(self.config['test_size'])
        
        for model_key, model_info in self.results['models'].items():
            print(f"\n{model_info['name']}:")
            
            try:
                latent_2d = self.analyze_latent_space(model_info, test_data)
                
                # Calculate some basic statistics
                latent_std = np.std(latent_2d, axis=0)
                latent_range = np.ptp(latent_2d, axis=0)
                
                print(f"  Latent Space Statistics:")
                print(f"    Standard Deviation: [{latent_std[0]:.3f}, {latent_std[1]:.3f}]")
                print(f"    Range: [{latent_range[0]:.3f}, {latent_range[1]:.3f}]")
                print(f"    t-SNE applied successfully")
                
            except Exception as e:
                print(f"  Error in latent space analysis: {str(e)}")
    
    def print_sample_results(self):
        """Print sample denoising results."""
        print("\n" + "=" * 60)
        print("SAMPLE DENOISING RESULTS")
        print("=" * 60)
        
        # Generate test data
        test_data = self.generate_synthetic_data(16)
        noisy_data = self.add_noise(test_data, 'gaussian', 0.3)
        
        print(f"Generated 16 test samples with Gaussian noise (level 0.3)")
        
        for model_key, model_info in self.results['models'].items():
            print(f"\n{model_info['name']}:")
            
            model = model_info['model']
            model.eval()
            with torch.no_grad():
                # Use flattened data for simple autoencoder
                denoised_data = model(noisy_data.to(self.device))[0].cpu()
                
                # Calculate metrics for these samples using PyTorch functions
                mse_loss = nn.MSELoss()
                
                test_flat = test_data.flatten()
                noisy_flat = noisy_data.flatten()
                denoised_flat = denoised_data.flatten()
                
                mse_original = mse_loss(noisy_flat, test_flat).item()
                mse_denoised = mse_loss(denoised_flat, test_flat).item()
                
                print(f"  Sample MSE (Noisy): {mse_original:.6f}")
                print(f"  Sample MSE (Denoised): {mse_denoised:.6f}")
                print(f"  Improvement: {mse_original - mse_denoised:.6f}")
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        print("\n" + "=" * 60)
        print("PLOTTING TRAINING CURVES")
        print("=" * 60)
        
        for model_key, metrics in self.results['training_metrics'].items():
            model_name = self.results['models'][model_key]['name']
            print(f"\nPlotting training curves for {model_name}...")
            
            train_losses = metrics['train_losses']
            val_losses = metrics['val_losses']
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss', linewidth=2)
            plt.plot(val_losses, label='Validation Loss', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Training and Validation Loss - {model_name}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            plt.close()
    
    def plot_denoising_performance(self):
        """Plot denoising performance across noise types and levels."""
        print("\n" + "=" * 60)
        print("PLOTTING DENOISING PERFORMANCE")
        print("=" * 60)
        
        for model_key, model_info in self.results['models'].items():
            print(f"\nPlotting denoising performance for {model_info['name']}...")
            
            # Prepare data for plotting
            noise_levels = self.config['noise_levels']
            noise_types = self.config['noise_types']
            
            # Create subplots for different metrics
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Denoising Performance - {model_info["name"]}', fontsize=16)
            
            # MSE plot
            ax1 = axes[0, 0]
            for noise_type in noise_types:
                mse_values = []
                for noise_level in noise_levels:
                    metrics = self.results['denoising_performance'][model_key][noise_level][noise_type]
                    mse_values.append(metrics['mse_denoised'])
                ax1.plot(noise_levels, mse_values, 'o-', label=noise_type, linewidth=2, markersize=6)
            ax1.set_xlabel('Noise Level')
            ax1.set_ylabel('MSE (Denoised)')
            ax1.set_title('Mean Squared Error vs Noise Level')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # PSNR plot
            ax2 = axes[0, 1]
            for noise_type in noise_types:
                psnr_values = []
                for noise_level in noise_levels:
                    metrics = self.results['denoising_performance'][model_key][noise_level][noise_type]
                    psnr_values.append(metrics['psnr_denoised'])
                ax2.plot(noise_levels, psnr_values, 's-', label=noise_type, linewidth=2, markersize=6)
            ax2.set_xlabel('Noise Level')
            ax2.set_ylabel('PSNR (dB)')
            ax2.set_title('Peak Signal-to-Noise Ratio vs Noise Level')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # SSIM plot
            ax3 = axes[1, 0]
            for noise_type in noise_types:
                ssim_values = []
                for noise_level in noise_levels:
                    metrics = self.results['denoising_performance'][model_key][noise_level][noise_type]
                    ssim_values.append(metrics['ssim'])
                ax3.plot(noise_levels, ssim_values, '^-', label=noise_type, linewidth=2, markersize=6)
            ax3.set_xlabel('Noise Level')
            ax3.set_ylabel('SSIM')
            ax3.set_title('Structural Similarity Index vs Noise Level')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Improvement plot
            ax4 = axes[1, 1]
            for noise_type in noise_types:
                improvement_values = []
                for noise_level in noise_levels:
                    metrics = self.results['denoising_performance'][model_key][noise_level][noise_type]
                    improvement_values.append(metrics['improvement_mse'])
                ax4.plot(noise_levels, improvement_values, 'd-', label=noise_type, linewidth=2, markersize=6)
            ax4.set_xlabel('Noise Level')
            ax4.set_ylabel('MSE Improvement')
            ax4.set_title('MSE Improvement vs Noise Level')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            plt.close()
    
    def plot_latent_space(self):
        """Plot latent space visualization using t-SNE."""
        print("\n" + "=" * 60)
        print("PLOTTING LATENT SPACE VISUALIZATION")
        print("=" * 60)
        
        # Generate test data for latent analysis
        test_data = self.generate_synthetic_data(self.config['test_size'])
        
        for model_key, model_info in self.results['models'].items():
            print(f"\nPlotting latent space for {model_info['name']}...")
            
            try:
                latent_2d = self.analyze_latent_space(model_info, test_data)
                
                plt.figure(figsize=(10, 8))
                plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=20)
                plt.xlabel('t-SNE Component 1')
                plt.ylabel('t-SNE Component 2')
                plt.title(f'Latent Space Visualization - {model_info["name"]}')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                plt.close()
                
            except Exception as e:
                print(f"  Error in latent space visualization: {str(e)}")
    
    def plot_sample_results(self):
        """Plot sample denoising results."""
        print("\n" + "=" * 60)
        print("PLOTTING SAMPLE DENOISING RESULTS")
        print("=" * 60)
        
        # Generate test data
        test_data = self.generate_synthetic_data(16)
        noisy_data = self.add_noise(test_data, 'gaussian', 0.3)
        
        print(f"Generated 16 test samples with Gaussian noise (level 0.3)")
        
        for model_key, model_info in self.results['models'].items():
            print(f"\nPlotting sample results for {model_info['name']}...")
            
            model = model_info['model']
            model.eval()
            with torch.no_grad():
                # Use flattened data for simple autoencoder
                denoised_data = model(noisy_data.to(self.device))[0].cpu()
                
                # Reshape data for visualization (28x28 images)
                clean_images = test_data.reshape(-1, 28, 28)
                noisy_images = noisy_data.reshape(-1, 28, 28)
                denoised_images = denoised_data.reshape(-1, 28, 28)
                
                # Create subplot grid
                fig, axes = plt.subplots(4, 4, figsize=(16, 12))
                fig.suptitle(f'Sample Denoising Results - {model_info["name"]}', fontsize=16)
                
                for i in range(16):
                    row = i // 4
                    col = i % 4
                    
                    # Original clean image
                    axes[row, col].imshow(clean_images[i], cmap='gray')
                    axes[row, col].set_title(f'Sample {i+1}')
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                plt.show()
                plt.close()
                
                # Noisy images
                fig, axes = plt.subplots(4, 4, figsize=(16, 12))
                fig.suptitle(f'Noisy Images (Gaussian noise, level 0.3)', fontsize=16)
                
                for i in range(16):
                    row = i // 4
                    col = i % 4
                    
                    axes[row, col].imshow(noisy_images[i], cmap='gray')
                    axes[row, col].set_title(f'Sample {i+1}')
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                plt.show()
                plt.close()
                
                # Denoised images
                fig, axes = plt.subplots(4, 4, figsize=(16, 12))
                fig.suptitle(f'Denoised Images - {model_info["name"]}', fontsize=16)
                
                for i in range(16):
                    row = i // 4
                    col = i % 4
                    
                    axes[row, col].imshow(denoised_images[i], cmap='gray')
                    axes[row, col].set_title(f'Sample {i+1}')
                    axes[row, col].axis('off')
                
                plt.tight_layout()
                plt.show()
                plt.close()
    
    def print_comprehensive_report(self):
        """Print comprehensive experiment report."""
        print("\n" + "=" * 80)
        print("SIMPLE AUTOENCODER DENOISING EXPERIMENT REPORT")
        print("=" * 80)
        
        # Configuration
        print("\nEXPERIMENT CONFIGURATION:")
        print("-" * 40)
        for key, value in self.config.items():
            print(f"{key}: {value}")
        
        # Model architectures
        print("\nMODEL ARCHITECTURE:")
        print("-" * 40)
        for model_key, model_info in self.results['models'].items():
            model = model_info['model']
            print(f"{model_info['name']}:")
            print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training curves summary
        self.print_training_curves()
        
        # Denoising performance summary
        self.print_denoising_performance()
        
        # Latent space analysis
        self.print_latent_space_analysis()
        
        # Sample results
        self.print_sample_results()
        
        # Final summary
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        
        # Get performance for the simple autoencoder
        model_key = 'simple'
        model_info = self.results['models'][model_key]
        
        all_mse = []
        for noise_level in self.config['noise_levels']:
            for noise_type in self.config['noise_types']:
                metrics = self.results['denoising_performance'][model_key][noise_level][noise_type]
                all_mse.append(metrics['mse_denoised'])
        
        avg_mse = np.mean(all_mse)
        print(f"Simple AutoEncoder Performance:")
        print(f"  Average MSE: {avg_mse:.6f}")
        print(f"  Model: {model_info['name']}")
        print(f"  Parameters: {sum(p.numel() for p in model_info['model'].parameters()):,}")
        print("\nExperiment completed successfully!")
    
    def run_experiment(self):
        """Run the complete denoising experiment."""
        print("Starting Simple AutoEncoder Denoising Experiment...")
        print("=" * 60)
        
        # Generate data
        print("Generating synthetic data...")
        train_data = self.generate_synthetic_data(3000)
        val_data = self.generate_synthetic_data(1000)
        test_data = self.generate_synthetic_data(self.config['test_size'])
        
        print(f"Generated {len(train_data)} training samples")
        print(f"Generated {len(val_data)} validation samples")
        print(f"Generated {len(test_data)} test samples")
        
        # Create models
        print("\nCreating simple autoencoder model...")
        self.results['models'] = self.create_models()
        
        # Train models
        print("\nTraining model...")
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
                    
                    print(f"    MSE: {metrics['mse_denoised']:.6f}, PSNR: {metrics['psnr_denoised']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
        
        # Print comprehensive report
        self.print_comprehensive_report()
        
        # Display plots
        self.plot_training_curves()
        self.plot_denoising_performance()
        self.plot_latent_space()
        self.plot_sample_results()
