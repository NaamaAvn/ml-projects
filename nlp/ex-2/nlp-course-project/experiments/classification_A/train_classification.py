#!/usr/bin/env python3
"""
Classification Training Script - Step 3: Training the Classification Model

This script implements step 3 of the text classification pipeline:
- Loads the classification model from step 2 (with frozen language model backbone)
- Sets up training data loaders for the classification task
- Trains the classification head using the pre-trained language model features
- Saves the trained model and training history

The language model parameters remain frozen during training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import argparse
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Import the classification model from step 2
from setup_classification_model import (
    LanguageModel, SimpleVocab, IMDBDataset,
    LanguageModelFeatureExtractor, ClassificationHead, ClassificationModel,
    load_pretrained_language_model, create_classification_model
)

class IMDBClassificationDataset(torch.utils.data.Dataset):
    """Custom Dataset for IMDB text classification."""
    def __init__(self, data_file: str, vocab: Any, max_length: int):
        """
        Args:
            data_file: Path to the JSON file containing processed sequences
            vocab: Vocabulary object for token to index conversion
            max_length: Maximum length of sequences (will be padded/truncated)
        """
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text, label = self.data[idx]
        
        # Convert text to tokens and then to indices
        tokens = text.split()
        indices = [self.vocab[token] for token in tokens]
        
        # Pad or truncate to max_length
        if len(indices) < self.max_length:
            # Pad with <pad> token (assuming <pad> has index 1)
            indices.extend([1] * (self.max_length - len(indices)))
        else:
            # Truncate to max_length
            indices = indices[:self.max_length]
        
        # Convert label to tensor (0 for negative, 1 for positive)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return torch.tensor(indices), label_tensor

def create_dataloaders(data_dir: str, vocab: SimpleVocab, batch_size: int = 32, 
                      max_length: int = 200) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation and test sets.
    
    Args:
        data_dir: Directory containing the processed classification data
        vocab: Vocabulary object
        batch_size: Batch size for training
        max_length: Maximum sequence length
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_file = os.path.join(data_dir, 'train.json')
    val_file = os.path.join(data_dir, 'val.json')
    test_file = os.path.join(data_dir, 'test.json')
    
    # Create datasets
    train_dataset = IMDBClassificationDataset(train_file, vocab, max_length)
    val_dataset = IMDBClassificationDataset(val_file, vocab, max_length)
    test_dataset = IMDBClassificationDataset(test_file, vocab, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"DataLoaders created:")
    print(f"  - Training samples: {len(train_dataset)}")
    print(f"  - Validation samples: {len(val_dataset)}")
    print(f"  - Test samples: {len(test_dataset)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Max sequence length: {max_length}")
    
    return train_loader, val_loader, test_loader

def train_epoch(model: ClassificationModel, train_loader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer, 
                device: str = 'cpu') -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: Classification model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to run training on
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (input_sequences, labels) in enumerate(progress_bar):
        # Move data to device
        input_sequences = input_sequences.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(input_sequences)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        
        # Calculate accuracy
        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        total_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100 * total_correct / total_samples:.2f}%'
        })
    
    average_loss = total_loss / len(train_loader)
    accuracy = total_correct / total_samples
    
    return average_loss, accuracy

def validate_epoch(model: ClassificationModel, val_loader: DataLoader, 
                  criterion: nn.Module, device: str = 'cpu') -> Tuple[float, float]:
    """
    Validate the model for one epoch.
    
    Args:
        model: Classification model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run validation on
        
    Returns:
        Tuple of (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, (input_sequences, labels) in enumerate(val_loader):
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
    
    average_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples
    
    return average_loss, accuracy

def train_classification_model(model: ClassificationModel, 
                             train_loader: DataLoader, 
                             val_loader: DataLoader,
                             epochs: int = 10,
                             learning_rate: float = 0.001,
                             device: str = 'cpu',
                             save_path: str = './classification_model.pth') -> Dict[str, List[float]]:
    """
    Train the classification model.
    
    Args:
        model: Classification model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to run training on
        save_path: Path to save the trained model
        
    Returns:
        Training history dictionary
    """
    print(f"Starting training for {epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    
    # Set up loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'history': history
            }, save_path)
            print(f"New best model saved! Validation accuracy: {val_acc:.4f}")
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return history

def main(model_dir: str = '../../model/',
         data_dir: str = '../../data/processed_classification_data/',
         output_dir: str = './',
         config_file: str = './classification_model_config.json',
         epochs: int = 10,
         batch_size: int = 32,
         learning_rate: float = 0.001,
         max_length: int = 200,
         device: str = 'cpu'):
    """
    Main function to train the classification model.
    
    Args:
        model_dir: Directory containing the pre-trained language model
        data_dir: Directory containing classification data
        output_dir: Directory to save outputs
        config_file: Path to the classification model config from step 2
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimization
        max_length: Maximum sequence length
        device: Device to run training on
    """
    print("Classification Training - Step 3: Training the Classification Model")
    print("=" * 80)
    
    # Check if required files exist
    model_path = os.path.join(model_dir, 'language_model.pth')
    if not os.path.exists(model_path):
        print(f"Error: Pre-trained language model not found at {model_path}")
        return
    
    if not os.path.exists(config_file):
        print(f"Error: Classification model config not found at {config_file}")
        print("Please run step 2 first to create the classification model.")
        return
    
    # Load configuration from step 2
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded configuration: {config}")
    
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
    
    # Create classification model using the same configuration as step 2
    print("Creating classification model...")
    classification_model = create_classification_model(
        language_model=language_model,
        hidden_dims=config['hidden_dims'],
        num_classes=config['num_classes'],
        pooling_method=config['pooling_method'],
        dropout=config['dropout']
    )
    
    # Move model to device
    classification_model = classification_model.to(device)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir, vocab, batch_size, max_length
    )
    
    # Train the model
    print("Starting training...")
    save_path = os.path.join(output_dir, 'trained_classification_model.pth')
    history = train_classification_model(
        model=classification_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        save_path=save_path
    )
    
    print(f"\nStep 3 completed successfully!")
    print(f"Trained model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification Training - Step 3")
    parser.add_argument('--model-dir', type=str, default='./model/',
                       help='Directory containing the pre-trained language model')
    parser.add_argument('--data-dir', type=str, default='./data/processed_classification_data/',
                       help='Directory containing classification data')
    parser.add_argument('--output-dir', type=str, default='./experiments/classification_A/',
                       help='Directory to save outputs')
    parser.add_argument('--config-file', type=str, default='./experiments/classification_A/classification_model_config.json',
                       help='Path to classification model config from step 2')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate for optimization')
    parser.add_argument('--max-length', type=int, default=200,
                       help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run training on')
    
    args = parser.parse_args()
    
    main(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config_file=args.config_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        device=args.device
    ) 