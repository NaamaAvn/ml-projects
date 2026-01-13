#!/usr/bin/env python3
"""
Classification Training Script - Step 3: Training the RNN Classification Model

This script implements step 3 of the text classification pipeline:
- Loads the RNN classification model from step 2 (with Word2Vec embeddings)
- Sets up training data loaders for the classification task
- Trains the complete RNN model end-to-end
- Saves the trained model and training history

This is an end-to-end training approach where both the RNN and classification head
are trained together, unlike Experiment A where the language model was frozen.
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

# Import the RNN classifier from step 2
from setup_classification_model import (
    SimpleVocab, IMDBDataset, RNNClassifier, create_rnn_classifier, evaluate_model
)

class IMDBClassificationDataset(torch.utils.data.Dataset):
    """Custom Dataset for IMDB text classification."""
    def __init__(self, data_file: str, vocab: SimpleVocab, max_length: int):
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
            # Pad with <pad> token
            indices.extend([self.vocab['<pad>']] * (self.max_length - len(indices)))
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

def train_epoch(model: RNNClassifier, train_loader: DataLoader, 
                criterion: nn.Module, optimizer: optim.Optimizer, 
                device: str = 'cpu') -> Tuple[float, float]:
    """
    Train the model for one epoch.
    
    Args:
        model: RNN classification model
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

def validate_epoch(model: RNNClassifier, val_loader: DataLoader, 
                  criterion: nn.Module, device: str = 'cpu') -> Tuple[float, float]:
    """
    Validate the model for one epoch.
    
    Args:
        model: RNN classification model
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

def train_rnn_classifier(model: RNNClassifier, 
                        train_loader: DataLoader, 
                        val_loader: DataLoader,
                        epochs: int = 10,
                        learning_rate: float = 0.001,
                        device: str = 'cpu',
                        save_path: str = './trained_rnn_classifier.pth') -> Dict[str, List[float]]:
    """
    Train the RNN classifier model.
    
    Args:
        model: RNN classification model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        device: Device to run training on
        save_path: Path to save the trained model
        
    Returns:
        Dictionary containing training history
    """
    print(f"Starting RNN classifier training for {epochs} epochs...")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {device}")
    
    # Loss function and optimizer
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
        'val_acc': [],
        'learning_rate': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    patience = 5
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 20)
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['learning_rate'].append(current_lr)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'history': history
            }, save_path)
            print(f"New best model saved! Validation accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            break
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    return history

def plot_training_history(history: Dict[str, List[float]], save_path: str = './training_history.png'):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot training and validation loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot training and validation accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to: {save_path}")

def main(model_dir: str = '../../model/',
         data_dir: str = '../../data/processed_classification_data/',
         output_dir: str = './',
         config_file: str = './rnn_classifier_config.json',
         epochs: int = 10,
         batch_size: int = 32,
         learning_rate: float = 0.001,
         max_length: int = 200,
         device: str = 'cpu'):
    """
    Main function to train the RNN classification model.
    
    Args:
        model_dir: Directory containing pre-trained models (not used in this experiment)
        data_dir: Directory containing processed classification data
        output_dir: Directory to save training results
        config_file: Path to the model configuration file
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimization
        max_length: Maximum sequence length
        device: Device to run training on
    """
    print("Training RNN Classification Model (Experiment B)")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model configuration
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded model configuration: {config}")
    
    # Load vocabulary
    vocab_file = os.path.join(os.path.dirname(config_file), config['vocab_file'])
    with open(vocab_file, 'r') as f:
        vocab_dict = json.load(f)
    
    vocab = SimpleVocab(vocab_dict)
    print(f"Loaded vocabulary with {len(vocab)} words")
    
    # Create RNN classifier
    model = create_rnn_classifier(
        vocab=vocab,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout'],
        rnn_type=config['rnn_type'],
        bidirectional=config['bidirectional'],
        device=device
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir, vocab, batch_size, max_length
    )
    
    # Train the model
    history = train_rnn_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
        save_path=os.path.join(output_dir, 'trained_rnn_classifier.pth')
    )
    
    # Plot training history
    plot_training_history(
        history, 
        save_path=os.path.join(output_dir, 'training_history.png')
    )
    
    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Save test results
    test_results = {
        'test_loss': test_loss,
        'test_accuracy': test_acc,
        'best_val_accuracy': max(history['val_acc'])
    }
    
    test_results_path = os.path.join(output_dir, 'test_results.json')
    with open(test_results_path, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"Test results saved to: {test_results_path}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RNN classification model with Word2Vec embeddings')
    parser.add_argument('--model-dir', type=str, default='../../model/',
                        help='Directory containing pre-trained models (not used in this experiment)')
    parser.add_argument('--data-dir', type=str, default='../../data/processed_classification_data/',
                        help='Directory containing processed classification data')
    parser.add_argument('--output-dir', type=str, default='./',
                        help='Directory to save training results')
    parser.add_argument('--config-file', type=str, default='./rnn_classifier_config.json',
                        help='Path to the model configuration file')
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