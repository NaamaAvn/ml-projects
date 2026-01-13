#!/usr/bin/env python3
"""
Classification Experiment A: Using Pre-trained Language Model as Feature Extractor

This script implements step 2 of the text classification pipeline:
- Uses the pre-trained language model as a "backbone" feature extractor
- The language model encodes input text into sentence embeddings
- A separate classification model (MLP) is added on top of these embeddings
- The classification model learns to classify text into predefined labels

This is a feature extraction approach where the language model weights are frozen
and only the classification head is trained.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import argparse
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Define the required classes directly to avoid import issues
class SimpleVocab:
    def __init__(self, stoi_dict):
        self.stoi = stoi_dict
        self.itos = {idx: token for token, idx in stoi_dict.items()}
        self._default_index = stoi_dict.get('', 0)
    
    def __getitem__(self, token):
        return self.stoi.get(token, self._default_index)
    
    def __len__(self):
        return len(self.stoi)

class LanguageModel(nn.Module):
    def __init__(self, vocab, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3, device='cpu'):
        super(LanguageModel, self).__init__()
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=vocab['<pad>'])
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, self.vocab_size)
        self.to(device)
    
    def forward(self, input_sequences, hidden=None):
        batch_size, seq_length = input_sequences.shape
        embeddings = self.embedding(input_sequences)
        embeddings = self.dropout(embeddings)
        lstm_output, (hidden_state, cell_state) = self.lstm(embeddings, hidden)
        lstm_output = self.dropout(lstm_output)
        output_logits = self.output_layer(lstm_output)
        return output_logits, (hidden_state, cell_state)

class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, vocab, seq_length):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.vocab = vocab
        self.seq_length = seq_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text, label = self.data[idx]
        tokens = text.split()
        indices = [self.vocab[token] for token in tokens]
        
        if len(indices) < self.seq_length:
            indices.extend([self.vocab['<pad>']] * (self.seq_length - len(indices)))
        else:
            indices = indices[:self.seq_length]
        
        return torch.tensor(indices), torch.tensor(label, dtype=torch.long)

class LanguageModelFeatureExtractor(nn.Module):
    """
    Feature extractor that uses a pre-trained language model to encode text.
    
    This class wraps the pre-trained language model and extracts sentence-level
    representations by using the final hidden states of the LSTM.
    """
    
    def __init__(self, language_model: LanguageModel, pooling_method: str = 'mean'):
        """
        Initialize the feature extractor.
        
        Args:
            language_model: Pre-trained language model
            pooling_method: Method to pool sequence representations ('mean', 'max', 'last')
        """
        super(LanguageModelFeatureExtractor, self).__init__()
        
        self.language_model = language_model
        self.pooling_method = pooling_method
        
        # Freeze the language model parameters
        for param in self.language_model.parameters():
            param.requires_grad = False
        
        # Get the hidden dimension from the language model
        self.feature_dim = language_model.hidden_dim
        
        print(f"Language Model Feature Extractor initialized:")
        print(f"  - Feature dimension: {self.feature_dim}")
        print(f"  - Pooling method: {pooling_method}")
        print(f"  - Language model parameters frozen: True")
    
    def forward(self, input_sequences: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input sequences using the language model.
        
        Args:
            input_sequences: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Sentence embeddings of shape (batch_size, feature_dim)
        """
        batch_size, seq_length = input_sequences.shape
        
        # Get the language model outputs
        with torch.no_grad():
            # Convert input indices to embeddings
            embeddings = self.language_model.embedding(input_sequences)
            
            # Apply dropout to embeddings
            embeddings = self.language_model.dropout(embeddings)
            
            # Pass through LSTM to get hidden states
            lstm_output, (hidden_state, _) = self.language_model.lstm(embeddings)
            # lstm_output shape: (batch_size, seq_length, hidden_dim)
            # hidden_state shape: (num_layers, batch_size, hidden_dim)
        
        # Extract sentence representations based on pooling method
        if self.pooling_method == 'mean':
            # Mean pooling over sequence length (excluding padding)
            # Create mask for non-padding tokens
            mask = (input_sequences != self.language_model.vocab['<pad>']).float()
            mask = mask.unsqueeze(-1)  # (batch_size, seq_length, 1)
            
            # Apply mask and compute mean
            masked_output = lstm_output * mask
            sentence_embeddings = masked_output.sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            
        elif self.pooling_method == 'max':
            # Max pooling over sequence length
            sentence_embeddings = torch.max(lstm_output, dim=1)[0]
            
        elif self.pooling_method == 'last':
            # Use the last hidden state from the top LSTM layer
            sentence_embeddings = hidden_state[-1]  # (batch_size, hidden_dim)
            
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling_method}")
        
        return sentence_embeddings

class ClassificationHead(nn.Module):
    """
    Classification head (MLP) that takes sentence embeddings and outputs class predictions.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], num_classes: int, dropout: float = 0.3):
        """
        Initialize the classification head.
        
        Args:
            input_dim: Dimension of input features (sentence embeddings)
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate for regularization
        """
        super(ClassificationHead, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        print(f"Classification Head initialized:")
        print(f"  - Input dimension: {input_dim}")
        print(f"  - Hidden dimensions: {hidden_dims}")
        print(f"  - Output classes: {num_classes}")
        print(f"  - Dropout rate: {dropout}")
    
    def forward(self, sentence_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classification head.
        
        Args:
            sentence_embeddings: Input features of shape (batch_size, input_dim)
            
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        return self.classifier(sentence_embeddings)

class ClassificationModel(nn.Module):
    """
    Complete classification model combining feature extractor and classification head.
    """
    
    def __init__(self, feature_extractor: LanguageModelFeatureExtractor, 
                 classification_head: ClassificationHead):
        """
        Initialize the classification model.
        
        Args:
            feature_extractor: Language model feature extractor
            classification_head: Classification head (MLP)
        """
        super(ClassificationModel, self).__init__()
        
        self.feature_extractor = feature_extractor
        self.classification_head = classification_head
        
        print(f"Classification Model initialized:")
        print(f"  - Feature extractor: {type(feature_extractor).__name__}")
        print(f"  - Classification head: {type(classification_head).__name__}")
    
    def forward(self, input_sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the complete classification model.
        
        Args:
            input_sequences: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # Extract features using the language model
        sentence_embeddings = self.feature_extractor(input_sequences)
        
        # Classify using the classification head
        class_logits = self.classification_head(sentence_embeddings)
        
        return class_logits
    
    def extract_features(self, input_sequences: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification (useful for analysis).
        
        Args:
            input_sequences: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Sentence embeddings of shape (batch_size, feature_dim)
        """
        return self.feature_extractor(input_sequences)

def load_pretrained_language_model(model_path: str, vocab: SimpleVocab, device: str = 'cpu') -> LanguageModel:
    """
    Load a pre-trained language model.
    
    Args:
        model_path: Path to the saved language model
        vocab: Vocabulary object
        device: Device to load the model on
        
    Returns:
        Loaded language model
    """
    # Load model configuration
    config_path = model_path.replace('.pth', '_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration if config file doesn't exist
        config = {
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.3
        }
    
    # Create language model with same configuration
    language_model = LanguageModel(
        vocab=vocab,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        device=device
    )
    
    # Load pre-trained weights
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            # Checkpoint format with model_state_dict
            language_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state_dict format
            language_model.load_state_dict(checkpoint)
    else:
        # Direct state_dict format
        language_model.load_state_dict(checkpoint)
    
    print(f"Pre-trained language model loaded from {model_path}")
    print(f"  - Model configuration: {config}")
    
    return language_model

def create_classification_model(language_model: LanguageModel, 
                              hidden_dims: List[int] = [256, 128], 
                              num_classes: int = 2,
                              pooling_method: str = 'mean',
                              dropout: float = 0.3) -> ClassificationModel:
    """
    Create a complete classification model.
    
    Args:
        language_model: Pre-trained language model
        hidden_dims: Hidden layer dimensions for classification head
        num_classes: Number of output classes
        pooling_method: Method to pool sequence representations
        dropout: Dropout rate for classification head
        
    Returns:
        Complete classification model
    """
    # Create feature extractor
    feature_extractor = LanguageModelFeatureExtractor(
        language_model=language_model,
        pooling_method=pooling_method
    )
    
    # Create classification head
    classification_head = ClassificationHead(
        input_dim=language_model.hidden_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout
    )
    
    # Create complete model
    classification_model = ClassificationModel(
        feature_extractor=feature_extractor,
        classification_head=classification_head
    )
    
    return classification_model

def evaluate_model(model: ClassificationModel, data_loader: DataLoader, 
                  criterion: nn.Module, device: str = 'cpu') -> Tuple[float, float]:
    """
    Evaluate the classification model.
    
    Args:
        model: Classification model
        data_loader: DataLoader for evaluation
        criterion: Loss function
        device: Device to run evaluation on
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (input_sequences, labels) in enumerate(data_loader):
            # Move data to device
            input_sequences = input_sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(input_sequences)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
    average_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples
    
    return average_loss, accuracy

def main(model_dir: str = '../../model/',
         data_dir: str = '../../data/processed_classification_data/',
         output_dir: str = './',
         pooling_method: str = 'mean',
         hidden_dims: List[int] = [256, 128],
         dropout: float = 0.3,
         device: str = 'cpu'):
    """
    Main function to demonstrate the classification model setup.
    
    Args:
        model_dir: Directory containing the pre-trained language model
        data_dir: Directory containing classification data
        output_dir: Directory to save outputs
        pooling_method: Method to pool sequence representations
        hidden_dims: Hidden layer dimensions for classification head
        dropout: Dropout rate for classification head
        device: Device to run the model on
    """
    print("Classification Experiment A: Using Pre-trained Language Model as Feature Extractor")
    print("=" * 80)
    
    # Check if required files exist
    model_path = os.path.join(model_dir, 'language_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Pre-trained language model not found at {model_path}")
        print("Please train the language model first using the main pipeline.")
        return
    
    # Load vocabulary
    vocab_path = os.path.join(model_dir.replace('model', 'data'), 'vocab.json')
    if not os.path.exists(vocab_path):
        print(f"Error: Vocabulary not found at {vocab_path}")
        return
    
    with open(vocab_path, 'r') as f:
        vocab_stoi = json.load(f)
    vocab = SimpleVocab(vocab_stoi)
    
    # Load pre-trained language model
    print("Loading pre-trained language model...")
    language_model = load_pretrained_language_model(model_path, vocab, device)
    
    # Create classification model
    print("Creating classification model...")
    classification_model = create_classification_model(
        language_model=language_model,
        hidden_dims=hidden_dims,
        num_classes=2,  # Binary classification for IMDB
        pooling_method=pooling_method,
        dropout=dropout
    )
    
    # Move model to device
    classification_model = classification_model.to(device)
    
    # Test the model with sample data
    print("Testing model with sample data...")
    batch_size = 4
    seq_length = 50
    sample_input = torch.randint(0, len(vocab), (batch_size, seq_length))
    sample_input = sample_input.to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = classification_model(sample_input)
        features = classification_model.extract_features(sample_input)
    
    print(f"Sample input shape: {sample_input.shape}")
    print(f"Extracted features shape: {features.shape}")
    print(f"Classification outputs shape: {outputs.shape}")
    print(f"Output probabilities: {torch.softmax(outputs, dim=1)}")
    
    # Save model configuration
    config = {
        'model_type': 'classification_with_lm_backbone',
        'pooling_method': pooling_method,
        'hidden_dims': hidden_dims,
        'num_classes': 2,
        'dropout': dropout,
        'feature_dim': language_model.hidden_dim,
        'vocab_size': len(vocab)
    }
    
    config_path = os.path.join(output_dir, 'classification_model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nModel configuration saved to {config_path}")
    print("\nStep 2 completed successfully!")
    print("The classification model is ready for training in the next step.")
    print("\nModel Summary:")
    print(f"  - Feature extractor: Language model with {pooling_method} pooling")
    print(f"  - Feature dimension: {language_model.hidden_dim}")
    print(f"  - Classification head: MLP with hidden dimensions {hidden_dims}")
    print(f"  - Output classes: 2 (binary classification)")
    print(f"  - Language model parameters: Frozen")
    print(f"  - Trainable parameters: Only classification head")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification Experiment A: LM as Feature Extractor")
    parser.add_argument('--model-dir', type=str, default='./model/',
                       help='Directory containing the pre-trained language model')
    parser.add_argument('--data-dir', type=str, default='./data/processed_data/',
                       help='Directory containing classification data')
    parser.add_argument('--output-dir', type=str, default='./experiments/classification_A/',
                       help='Directory to save outputs')
    parser.add_argument('--pooling-method', type=str, default='mean',
                       choices=['mean', 'max', 'last'],
                       help='Method to pool sequence representations')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128],
                       help='Hidden layer dimensions for classification head')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate for classification head')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run the model on')
    
    args = parser.parse_args()
    
    main(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pooling_method=args.pooling_method,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        device=args.device
    ) 