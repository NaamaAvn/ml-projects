import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional, Union


class ConvLayerConfig:
    """Configuration for a convolutional layer."""
    
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 pool_size: Optional[int] = None,
                 pool_stride: Optional[int] = None):
        """
        Initialize convolutional layer configuration.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Kernel size for convolution
            stride: Stride for convolution
            padding: Padding for convolution
            pool_size: Pooling kernel size (None = no pooling)
            pool_stride: Pooling stride (None = same as pool_size)
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_size = pool_size
        self.pool_stride = pool_stride if pool_stride is not None else pool_size


class ConvAutoEncoder(nn.Module):
    """
    AutoEncoder with configurable convolutional layers.
    
    Architecture:
    - Input -> Multiple Conv layers (encoder)
    - Latent layer (flattened)
    - Multiple Conv layers -> Output (decoder)
    """
    
    def __init__(self, 
                 input_shape: Tuple[int, int, int],  # (channels, height, width)
                 conv_configs: List[ConvLayerConfig],
                 latent_dim: int,
                 activation: str = 'relu',
                 dropout_rate: float = 0.0,
                 batch_norm: bool = False,
                 use_upsampling: bool = True):
        """
        Initialize the convolutional autoencoder.
        
        Args:
            input_shape: Shape of input (channels, height, width)
            conv_configs: List of convolutional layer configurations
            latent_dim: Dimension of latent space
            activation: Activation function ('relu', 'tanh', 'sigmoid', 'leaky_relu')
            dropout_rate: Dropout rate for regularization
            batch_norm: Whether to use batch normalization
            use_upsampling: Whether to use upsampling in decoder (True) or transposed conv (False)
        """
        super(ConvAutoEncoder, self).__init__()
        
        self.input_shape = input_shape
        self.conv_configs = conv_configs
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.use_upsampling = use_upsampling
        
        # Calculate the size after encoder convolutions
        self.encoder_output_shape = self._calculate_encoder_output_shape()
        
        # Build encoder
        self.encoder = self._build_encoder()
        
        # Calculate flattened size for latent layer
        self.flattened_size = self.encoder_output_shape[0] * self.encoder_output_shape[1] * self.encoder_output_shape[2]
        
        # Latent layer
        self.latent_layer = nn.Linear(self.flattened_size, latent_dim)
        self.latent_decoder = nn.Linear(latent_dim, self.flattened_size)
        
        # Build decoder
        self.decoder = self._build_decoder()
        
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
    
    def _calculate_encoder_output_shape(self) -> Tuple[int, int, int]:
        """Calculate the output shape after encoder convolutions."""
        channels, height, width = self.input_shape
        
        for config in self.conv_configs:
            # Convolution
            height = (height + 2 * config.padding - config.kernel_size) // config.stride + 1
            width = (width + 2 * config.padding - config.kernel_size) // config.stride + 1
            channels = config.out_channels
            
            # Pooling
            if config.pool_size is not None:
                height = height // config.pool_size
                width = width // config.pool_size
        
        return (channels, height, width)
    
    def _build_encoder(self) -> nn.ModuleList:
        """Build the encoder layers."""
        layers = nn.ModuleList()
        in_channels = self.input_shape[0]
        
        for config in self.conv_configs:
            # Convolutional layer
            conv_layer = nn.Conv2d(
                in_channels, config.out_channels, 
                kernel_size=config.kernel_size, 
                stride=config.stride, 
                padding=config.padding
            )
            layers.append(conv_layer)
            
            # Batch normalization
            if self.batch_norm:
                layers.append(nn.BatchNorm2d(config.out_channels))
            
            # Pooling layer
            if config.pool_size is not None:
                layers.append(nn.MaxPool2d(config.pool_size, config.pool_stride))
            
            in_channels = config.out_channels
        
        return layers
    
    def _build_decoder(self) -> nn.ModuleList:
        """Build the decoder layers."""
        layers = nn.ModuleList()
        
        # Reverse the conv configs for decoder
        reversed_configs = list(reversed(self.conv_configs))
        
        for i, config in enumerate(reversed_configs):
            # Upsampling or transposed convolution
            if self.use_upsampling and config.pool_size is not None:
                layers.append(nn.Upsample(
                    scale_factor=config.pool_size, 
                    mode='nearest'
                ))
            
            # Transposed convolution (or regular conv if using upsampling)
            if self.use_upsampling:
                # Use regular convolution after upsampling
                conv_layer = nn.Conv2d(
                    config.out_channels, 
                    config.in_channels if i < len(reversed_configs) - 1 else self.input_shape[0],
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    padding=config.padding
                )
            else:
                # Use transposed convolution
                conv_layer = nn.ConvTranspose2d(
                    config.out_channels,
                    config.in_channels if i < len(reversed_configs) - 1 else self.input_shape[0],
                    kernel_size=config.kernel_size,
                    stride=config.stride,
                    padding=config.padding
                )
            
            layers.append(conv_layer)
            
            # Batch normalization (except for last layer)
            if self.batch_norm and i < len(reversed_configs) - 1:
                layers.append(nn.BatchNorm2d(
                    config.in_channels if i < len(reversed_configs) - 1 else self.input_shape[0]
                ))
        
        return layers
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        # Apply encoder layers
        for layer in self.encoder:
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
                x = self.activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)
            elif isinstance(layer, nn.BatchNorm2d):
                x = layer(x)
            elif isinstance(layer, nn.MaxPool2d):
                x = layer(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Latent layer
        z = self.latent_layer(x)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output."""
        # Decode from latent
        x = self.latent_decoder(z)
        
        # Reshape to encoder output shape
        x = x.view(x.size(0), *self.encoder_output_shape)
        
        # Apply decoder layers
        for i, layer in enumerate(self.decoder):
            if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
                x = layer(x)
                # Don't apply activation to the last layer
                if i < len(self.decoder) - 1:
                    x = self.activation(x)
                    if self.dropout is not None:
                        x = self.dropout(x)
            elif isinstance(layer, nn.BatchNorm2d):
                x = layer(x)
            elif isinstance(layer, nn.Upsample):
                x = layer(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the autoencoder.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Tuple of (reconstructed_output, latent_representation)
        """
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z
    
    def get_encoder_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoder features before flattening."""
        for layer in self.encoder:
            if isinstance(layer, nn.Conv2d):
                x = layer(x)
                x = self.activation(x)
                if self.dropout is not None:
                    x = self.dropout(x)
            elif isinstance(layer, nn.BatchNorm2d):
                x = layer(x)
            elif isinstance(layer, nn.MaxPool2d):
                x = layer(x)
        return x


class ConvAutoEncoderTrainer:
    """Trainer class for the convolutional autoencoder."""
    
    def __init__(self, 
                 model: ConvAutoEncoder, 
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.0):
        """
        Initialize the trainer.
        
        Args:
            model: ConvAutoEncoder instance
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
            batch: Input batch of shape (batch_size, channels, height, width)
            
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
            data: Input data of shape (batch_size, channels, height, width)
            
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
            data: Input data of shape (batch_size, channels, height, width)
            
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
        Get encoder features (before flattening) for given data.
        
        Args:
            data: Input data of shape (batch_size, channels, height, width)
            
        Returns:
            Encoder features
        """
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            features = self.model.get_encoder_features(data)
            return features.cpu()


def create_conv_autoencoder(input_shape: Tuple[int, int, int],
                          conv_configs: List[ConvLayerConfig],
                          latent_dim: int,
                          activation: str = 'relu',
                          dropout_rate: float = 0.0,
                          batch_norm: bool = False,
                          use_upsampling: bool = True) -> ConvAutoEncoder:
    """
    Factory function to create a convolutional autoencoder.
    
    Args:
        input_shape: Shape of input (channels, height, width)
        conv_configs: List of convolutional layer configurations
        latent_dim: Dimension of latent space
        activation: Activation function
        dropout_rate: Dropout rate
        batch_norm: Whether to use batch normalization
        use_upsampling: Whether to use upsampling in decoder
        
    Returns:
        ConvAutoEncoder instance
    """
    return ConvAutoEncoder(
        input_shape, conv_configs, latent_dim, activation, 
        dropout_rate, batch_norm, use_upsampling
    )


# Example usage
if __name__ == "__main__":
    # Example parameters for MNIST-like images (28x28, 1 channel)
    input_shape = (1, 28, 28)  # (channels, height, width)
    latent_dim = 64
    
    # Define convolutional layer configurations
    conv_configs = [
        ConvLayerConfig(1, 32, kernel_size=3, pool_size=2),      # 28x28 -> 14x14
        ConvLayerConfig(32, 64, kernel_size=3, pool_size=2),     # 14x14 -> 7x7
        ConvLayerConfig(64, 128, kernel_size=3, pool_size=1),    # 7x7 -> 7x7
    ]
    
    # Create model
    model = create_conv_autoencoder(
        input_shape=input_shape,
        conv_configs=conv_configs,
        latent_dim=latent_dim,
        activation='relu',
        dropout_rate=0.2,
        batch_norm=True,
        use_upsampling=True
    )
    
    # Create trainer
    trainer = ConvAutoEncoderTrainer(model, learning_rate=0.001, weight_decay=1e-5)
    
    # Example training loop
    print("Convolutional AutoEncoder created successfully!")
    print(f"Input shape: {input_shape}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Dropout rate: {0.2}")
    print(f"Batch normalization: {True}")
    print(f"Use upsampling: {True}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Print model architecture
    print("\nModel architecture:")
    print(model)
    
    # Test with dummy data
    dummy_input = torch.randn(4, *input_shape)
    print(f"\nDummy input shape: {dummy_input.shape}")
    
    with torch.no_grad():
        recon, latent = model(dummy_input)
        print(f"Reconstructed output shape: {recon.shape}")
        print(f"Latent representation shape: {latent.shape}")
