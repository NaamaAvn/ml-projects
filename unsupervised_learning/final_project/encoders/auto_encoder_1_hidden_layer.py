import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SimpleAutoEncoder(nn.Module):
    """
    Simple AutoEncoder with one hidden layer.
    
    Architecture:
    - Input layer -> Hidden layer (encoder)
    - Hidden layer -> Output layer (decoder)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, activation: str = 'relu'):
        """
        Initialize the simple autoencoder.
        
        Args:
            input_dim: Dimension of input data
            hidden_dim: Dimension of hidden layer (latent space)
            activation: Activation function ('relu', 'tanh', 'sigmoid')
        """
        super(SimpleAutoEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder: input -> hidden
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # Decoder: hidden -> output
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        # Activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.activation(self.encoder(x))
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        return self.decoder(z)
    
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


class SimpleAutoEncoderTrainer:
    """Trainer class for the simple autoencoder."""
    
    def __init__(self, model: SimpleAutoEncoder, learning_rate: float = 0.001):
        """
        Initialize the trainer.
        
        Args:
            model: SimpleAutoEncoder instance
            learning_rate: Learning rate for optimization
        """
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
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
            Latent representations of shape (batch_size, hidden_dim)
        """
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            z = self.model.encode(data)
            return z.cpu()


def create_simple_autoencoder(input_dim: int, hidden_dim: int, activation: str = 'relu') -> SimpleAutoEncoder:
    """
    Factory function to create a simple autoencoder.
    
    Args:
        input_dim: Dimension of input data
        hidden_dim: Dimension of hidden layer
        activation: Activation function
        
    Returns:
        SimpleAutoEncoder instance
    """
    return SimpleAutoEncoder(input_dim, hidden_dim, activation)


# Example usage
if __name__ == "__main__":
    # Example parameters
    input_dim = 784  # e.g., flattened 28x28 image
    hidden_dim = 64
    batch_size = 32
    
    # Create model
    model = create_simple_autoencoder(input_dim, hidden_dim, activation='relu')
    
    # Create trainer
    trainer = SimpleAutoEncoderTrainer(model, learning_rate=0.001)
    
    # Example training loop
    print("Simple AutoEncoder with one hidden layer created successfully!")
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimension: {hidden_dim}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
