"""
AutoEncoder implementations package.

This package contains three different autoencoder architectures:
1. Simple AutoEncoder with one hidden layer
2. Multi-layer AutoEncoder with configurable hidden layers  
3. Convolutional AutoEncoder with configurable conv layers
"""

from .auto_encoder_1_hidden_layer import SimpleAutoEncoder, SimpleAutoEncoderTrainer, create_simple_autoencoder
from .auto_encoder_many_hidden_layers import MultiLayerAutoEncoder, MultiLayerAutoEncoderTrainer, create_multi_layer_autoencoder
from .auto_encoder_many_conv_layers import ConvAutoEncoder, ConvAutoEncoderTrainer, create_conv_autoencoder, ConvLayerConfig

__all__ = [
    'SimpleAutoEncoder',
    'SimpleAutoEncoderTrainer', 
    'create_simple_autoencoder',
    'MultiLayerAutoEncoder',
    'MultiLayerAutoEncoderTrainer',
    'create_multi_layer_autoencoder',
    'ConvAutoEncoder',
    'ConvAutoEncoderTrainer',
    'create_conv_autoencoder',
    'ConvLayerConfig'
] 