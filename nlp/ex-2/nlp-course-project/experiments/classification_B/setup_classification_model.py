#!/usr/bin/env python3
"""
Classification Experiment B: Using From-Scratch RNN with Pre-trained Word2Vec Embeddings

This script implements step 2 of the text classification pipeline:
- Uses a from-scratch RNN model (LSTM/GRU) with pre-trained Word2Vec embeddings
- The RNN processes input text and learns to classify text into predefined labels
- This is an end-to-end approach where both the RNN and classification head are trained together

The key difference from Experiment A is that we're not using a pre-trained language model
as a feature extractor, but instead training a complete RNN-based classifier from scratch.
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
import gensim.downloader as gensim_downloader
from gensim.models import KeyedVectors
import pickle

# Define the required classes directly to avoid import issues
class SimpleVocab:
    def __init__(self, stoi_dict):
        self.stoi = stoi_dict
        self.itos = {idx: token for token, idx in stoi_dict.items()}
        self._default_index = stoi_dict.get('<unk>', 0)
    
    def __getitem__(self, token):
        return self.stoi.get(token, self._default_index)
    
    def __len__(self):
        return len(self.stoi)

class Word2VecEmbeddingLayer(nn.Module):
    """
    Embedding layer that uses pre-trained Word2Vec vectors.
    """
    def __init__(self, vocab: SimpleVocab, embedding_dim: int = 300, freeze_embeddings: bool = False):
        super(Word2VecEmbeddingLayer, self).__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.freeze_embeddings = freeze_embeddings
        
        # Initialize embedding layer
        self.embedding = nn.Embedding(len(vocab), embedding_dim, padding_idx=vocab['<pad>'])
        
        # Load pre-trained Word2Vec vectors
        self._load_word2vec_embeddings()
        
        # Freeze embeddings if requested
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False
            print("Word2Vec embeddings frozen")
        else:
            print("Word2Vec embeddings are trainable")
    
    def _load_word2vec_embeddings(self):
        """Load pre-trained Word2Vec vectors and initialize embedding layer."""
        print("Loading pre-trained Word2Vec vectors...")
        
        try:
            # Try to load from local cache first
            cache_path = './word2vec_cache.pkl'
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    word2vec_model = pickle.load(f)
                print("Loaded Word2Vec from cache")
            else:
                # Download and load Word2Vec model
                print("Downloading Word2Vec model (this may take a while)...")
                word2vec_model = gensim_downloader.load('word2vec-google-news-300')
                
                # Cache the model
                with open(cache_path, 'wb') as f:
                    pickle.dump(word2vec_model, f)
                print("Word2Vec model cached")
            
            # Initialize embedding weights
            embedding_weights = torch.randn(len(self.vocab), self.embedding_dim) * 0.1
            
            # Set weights for known words
            found_words = 0
            for word, idx in self.vocab.stoi.items():
                if word in word2vec_model:
                    embedding_weights[idx] = torch.FloatTensor(word2vec_model[word])
                    found_words += 1
                elif word in ['<pad>', '<unk>', '<sos>', '<eos>']:
                    # Initialize special tokens with small random values
                    embedding_weights[idx] = torch.randn(self.embedding_dim) * 0.1
            
            # Set the embedding weights
            self.embedding.weight.data.copy_(embedding_weights)
            
            print(f"Word2Vec embeddings loaded: {found_words}/{len(self.vocab)} words found")
            
        except Exception as e:
            print(f"Warning: Could not load Word2Vec embeddings: {e}")
            print("Using random initialization instead")
    
    def forward(self, input_sequences: torch.Tensor) -> torch.Tensor:
        """Forward pass through the embedding layer."""
        return self.embedding(input_sequences)

class RNNClassifier(nn.Module):
    """
    From-scratch RNN classifier using pre-trained Word2Vec embeddings.
    """
    def __init__(self, vocab: SimpleVocab, embedding_dim: int = 300, hidden_dim: int = 256, 
                 num_layers: int = 2, num_classes: int = 2, dropout: float = 0.3, 
                 rnn_type: str = 'lstm', bidirectional: bool = True, device: str = 'cpu'):
        super(RNNClassifier, self).__init__()
        
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.device = device
        
        # Calculate output dimension based on bidirectional setting
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Embedding layer with Word2Vec
        self.embedding = Word2VecEmbeddingLayer(vocab, embedding_dim, freeze_embeddings=False)
        
        # RNN layer
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional,
                batch_first=True
            )
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        self.to(device)
        
        print(f"RNN Classifier initialized:")
        print(f"  - RNN type: {rnn_type.upper()}")
        print(f"  - Bidirectional: {bidirectional}")
        print(f"  - Hidden dimension: {hidden_dim}")
        print(f"  - Number of layers: {num_layers}")
        print(f"  - Output dimension: {self.output_dim}")
        print(f"  - Number of classes: {num_classes}")
    
    def forward(self, input_sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the RNN classifier.
        
        Args:
            input_sequences: Input tensor of shape (batch_size, seq_length)
            
        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        batch_size, seq_length = input_sequences.shape
        
        # Get embeddings
        embeddings = self.embedding(input_sequences)  # (batch_size, seq_length, embedding_dim)
        embeddings = self.dropout(embeddings)
        
        # Pack padded sequences for better efficiency
        lengths = (input_sequences != self.vocab['<pad>']).sum(dim=1)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Pass through RNN
        packed_output, (hidden, cell) = self.rnn(packed_embeddings)
        
        # Unpack the output
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Get the final hidden state for classification
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            if self.rnn_type == 'lstm':
                final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch_size, hidden_dim * 2)
            else:  # GRU
                final_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch_size, hidden_dim * 2)
        else:
            # Use the final hidden state from the last layer
            final_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Apply dropout
        final_hidden = self.dropout(final_hidden)
        
        # Classification
        logits = self.classifier(final_hidden)
        
        return logits

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

def create_vocab_from_data(data_dir: str, min_freq: int = 2) -> SimpleVocab:
    """
    Create vocabulary from the training data.
    
    Args:
        data_dir: Directory containing the processed data
        min_freq: Minimum frequency for a word to be included in vocabulary
        
    Returns:
        SimpleVocab object
    """
    print("Creating vocabulary from training data...")
    
    # Load training data
    train_file = os.path.join(data_dir, 'train.json')
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    
    # Count word frequencies
    word_freq = {}
    for text, _ in train_data:
        tokens = text.split()
        for token in tokens:
            word_freq[token] = word_freq.get(token, 0) + 1
    
    # Filter by minimum frequency
    filtered_words = [word for word, freq in word_freq.items() if freq >= min_freq]
    
    # Add special tokens
    special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
    vocab_words = special_tokens + sorted(filtered_words)
    
    # Create vocabulary
    stoi = {word: idx for idx, word in enumerate(vocab_words)}
    vocab = SimpleVocab(stoi)
    
    print(f"Vocabulary created: {len(vocab)} words (min_freq={min_freq})")
    return vocab

def create_rnn_classifier(vocab: SimpleVocab, 
                         embedding_dim: int = 300,
                         hidden_dim: int = 256,
                         num_layers: int = 2,
                         num_classes: int = 2,
                         dropout: float = 0.3,
                         rnn_type: str = 'lstm',
                         bidirectional: bool = True,
                         device: str = 'cpu') -> RNNClassifier:
    """
    Create an RNN classifier with Word2Vec embeddings.
    
    Args:
        vocab: Vocabulary object
        embedding_dim: Dimension of Word2Vec embeddings
        hidden_dim: Hidden dimension of RNN
        num_layers: Number of RNN layers
        num_classes: Number of output classes
        dropout: Dropout rate
        rnn_type: Type of RNN ('lstm' or 'gru')
        bidirectional: Whether to use bidirectional RNN
        device: Device to run on
        
    Returns:
        RNNClassifier model
    """
    model = RNNClassifier(
        vocab=vocab,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout=dropout,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        device=device
    )
    
    return model

def evaluate_model(model: RNNClassifier, data_loader: DataLoader, 
                  criterion: nn.Module, device: str = 'cpu') -> Tuple[float, float]:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: RNN classifier model
        data_loader: Data loader for evaluation
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
         embedding_dim: int = 300,
         hidden_dim: int = 256,
         num_layers: int = 2,
         dropout: float = 0.3,
         rnn_type: str = 'lstm',
         bidirectional: bool = True,
         min_freq: int = 2,
         device: str = 'cpu'):
    """
    Main function to set up the RNN classification model.
    
    Args:
        model_dir: Directory containing pre-trained models (not used in this experiment)
        data_dir: Directory containing processed classification data
        output_dir: Directory to save the model configuration
        embedding_dim: Dimension of Word2Vec embeddings
        hidden_dim: Hidden dimension of RNN
        num_layers: Number of RNN layers
        dropout: Dropout rate
        rnn_type: Type of RNN ('lstm' or 'gru')
        bidirectional: Whether to use bidirectional RNN
        min_freq: Minimum frequency for vocabulary creation
        device: Device to run on
    """
    print("Setting up RNN Classification Model (Experiment B)")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create vocabulary from training data
    vocab = create_vocab_from_data(data_dir, min_freq=min_freq)
    
    # Create RNN classifier
    model = create_rnn_classifier(
        vocab=vocab,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=2,  # Binary classification for IMDB
        dropout=dropout,
        rnn_type=rnn_type,
        bidirectional=bidirectional,
        device=device
    )
    
    # Save vocabulary
    vocab_path = os.path.join(output_dir, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab.stoi, f, indent=2)
    print(f"Vocabulary saved to: {vocab_path}")
    
    # Save model configuration
    config = {
        'model_type': 'rnn_classifier',
        'vocab_size': len(vocab),
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'num_classes': 2,
        'dropout': dropout,
        'rnn_type': rnn_type,
        'bidirectional': bidirectional,
        'min_freq': min_freq,
        'vocab_file': 'vocab.json'
    }
    
    config_path = os.path.join(output_dir, 'rnn_classifier_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Model configuration saved to: {config_path}")
    
    # Save the model
    model_path = os.path.join(output_dir, 'trained_rnn_classifier.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'vocab': vocab.stoi
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    print("\nSetup complete!")
    print(f"Model configuration: {config}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set up RNN classification model with Word2Vec embeddings')
    parser.add_argument('--model-dir', type=str, default='../../model/',
                        help='Directory containing pre-trained models (not used in this experiment)')
    parser.add_argument('--data-dir', type=str, default='../../data/processed_classification_data/',
                        help='Directory containing processed classification data')
    parser.add_argument('--output-dir', type=str, default='./',
                        help='Directory to save the model configuration')
    parser.add_argument('--embedding-dim', type=int, default=300,
                        help='Dimension of Word2Vec embeddings')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension of RNN')
    parser.add_argument('--num-layers', type=int, default=2,
                        help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate')
    parser.add_argument('--rnn-type', type=str, default='lstm', choices=['lstm', 'gru'],
                        help='Type of RNN to use')
    parser.add_argument('--bidirectional', action='store_true', default=True,
                        help='Whether to use bidirectional RNN')
    parser.add_argument('--min-freq', type=int, default=2,
                        help='Minimum frequency for vocabulary creation')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run on')
    
    args = parser.parse_args()
    
    main(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        rnn_type=args.rnn_type,
        bidirectional=args.bidirectional,
        min_freq=args.min_freq,
        device=args.device
    ) 