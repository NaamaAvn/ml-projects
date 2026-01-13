import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


class MultiLayerAutoEncoder(nn.Module):
    """
    AutoEncoder with configurable multiple hidden layers.
    
    Architecture:
    - Input layer -> Multiple hidden layers (encoder)
    - Latent layer
    - Multiple hidden layers -> Output layer (decoder)
    """
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dims: List[int], 
                 latent_dim: int,
                 activation: str = 'relu',
                 dropout_rate: float = 0.0,
                 batch_norm: bool = False):
        """
        Initialize the multi-layer autoencoder.
        
        Args:
            input_dim: Dimension of input data
            hidden_dims: List of hidden layer dimensions for encoder/decoder
            latent_dim: Dimension of latent space
            activation: Activation function ('relu', 'tanh', 'sigmoid', 'leaky_relu')
            dropout_rate: Dropout rate for regularization (0.0 = no dropout)
            batch_norm: Whether to use batch normalization
        """
        super(MultiLayerAutoEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        
        # Build encoder layers
        self.encoder_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layer = nn.Linear(prev_dim, hidden_dim)
            self.encoder_layers.append(layer)
            
            if batch_norm:
                self.encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            
            prev_dim = hidden_dim
        
        # Latent layer
        self.latent_layer = nn.Linear(prev_dim, latent_dim)
        
        # Build decoder layers (reverse of encoder)
        self.decoder_layers = nn.ModuleList()
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            layer = nn.Linear(prev_dim, hidden_dim)
            self.decoder_layers.append(layer)
            
            if batch_norm:
                self.decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, input_dim)
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        for i, layer in enumerate(self.encoder_layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                x = self.activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
        
        # Latent layer
        z = self.latent_layer(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        x = z
        
        for i, layer in enumerate(self.decoder_layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                x = self.activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
        
        # Output layer
        x_recon = self.output_layer(x)
        return x_recon
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (reconstructed_output, latent_representation)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def get_encoder_output(self, x: torch.Tensor) -> torch.Tensor:
        """Get the output of the encoder before the latent layer."""
        for i, layer in enumerate(self.encoder_layers):
            if isinstance(layer, nn.Linear):
                x = layer(x)
                x = self.activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
        return x


class MultiLayerAutoEncoderTrainer:
    """Trainer class for the multi-layer autoencoder."""
    
    def __init__(self, 
                 model: MultiLayerAutoEncoder, 
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0):
        """
        Initialize the trainer.
        
        Args:
            model: MultiLayerAutoEncoder instance
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
        """
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train_step(self, batch: torch.Tensor) -> float:
        """
        Perform one training step.
        
        Args:
            batch: Input batch of shape (batch_size, input_dim)
            
        Returns:
            Loss value
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        batch = batch.to(self.device)
        x_recon, _ = self.model(batch)
        
        loss = self.criterion(x_recon, batch)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def evaluate(self, data: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Evaluate the model on given data.
        
        Args:
            data: Input data of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (reconstruction_loss, latent_representations)
        """
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            x_recon, z = self.model(data)
            loss = self.criterion(x_recon, data)
            return loss.item(), z.cpu()
    
    def get_latent_representations(self, data: torch.Tensor) -> torch.Tensor:
        """
        Get latent representations for given data.
        
        Args:
            data: Input data of shape (batch_size, input_dim)
            
        Returns:
            Latent representations of shape (batch_size, latent_dim)
        """
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            z = self.model.encode(data)
            return z.cpu()
    
    def get_encoder_features(self, data: torch.Tensor) -> torch.Tensor:
        """
        Get encoder features (before latent layer) for given data.
        
        Args:
            data: Input data of shape (batch_size, input_dim)
            
        Returns:
            Encoder features of shape (batch_size, last_hidden_dim)
        """
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            features = self.model.get_encoder_output(data)
            return features.cpu()


def create_multi_layer_autoencoder(input_dim: int, 
                                 hidden_dims: List[int], 
                                 latent_dim: int,
                                 activation: str = 'relu',
                                 dropout_rate: float = 0.0,
                                 batch_norm: bool = False) -> MultiLayerAutoEncoder:
    """
    Factory function to create a multi-layer autoencoder.
    
    Args:
        input_dim: Dimension of input data
        hidden_dims: List of hidden layer dimensions
        latent_dim: Dimension of latent space
        activation: Activation function
        dropout_rate: Dropout rate
        batch_norm: Whether to use batch normalization
        
    Returns:
        MultiLayerAutoEncoder instance
    """
    return MultiLayerAutoEncoder(input_dim, hidden_dims, latent_dim, activation, dropout_rate, batch_norm)


# Example usage
if __name__ == "__main__":
    # Example parameters
    input_dim = 784  # e.g., flattened 28x28 image
    hidden_dims = [512, 256, 128]  # Multiple hidden layers
    latent_dim = 64
    dropout_rate = 0.2
    batch_norm = True
    
    # Create model
    model = create_multi_layer_autoencoder(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        activation='relu',
        dropout_rate=dropout_rate,
        batch_norm=batch_norm
    )
    
    # Create trainer
    trainer = MultiLayerAutoEncoderTrainer(model, learning_rate=0.001, weight_decay=1e-5)
    
    # Example training loop
    print("Multi-layer AutoEncoder created successfully!")
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimensions: {hidden_dims}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Dropout rate: {dropout_rate}")
    print(f"Batch normalization: {batch_norm}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Print model architecture
    print("\nModel architecture:")
    print(model)
