import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from typing import Dict, List, Tuple, Any

# Import the LanguageModel class and related functions from the original script
class LanguageModel(nn.Module):
    """
    Language Model using LSTM architecture for next word prediction.
    
    This model consists of:
    1. Embedding layer to convert token indices to dense vectors
    2. LSTM layers to capture sequential dependencies
    3. Dropout for regularization
    4. Output layer to predict probability distribution over vocabulary
    """
    
    def __init__(self, vocab: Any, embedding_dim: int = 100, hidden_dim: int = 100, 
                 num_layers: int = 2, dropout: float = 0.2, device: str = 'cpu'):
        """
        Initialize the Language Model.
        
        Args:
            vocab: Vocabulary object for token to index conversion
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout rate for regularization
            device: Device to run the model on ('cpu' or 'cuda')
        """
        super(LanguageModel, self).__init__()
        
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.device = device
        
        # Embedding layer
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=vocab['<pad>'])
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output layer to predict next word
        self.output_layer = nn.Linear(hidden_dim, self.vocab_size)
        
        # Move model to device
        self.to(device)
        
        print(f"Language Model initialized:")
        print(f"  - Vocabulary size: {self.vocab_size}")
        print(f"  - Embedding dimension: {embedding_dim}")
        print(f"  - Hidden dimension: {hidden_dim}")
        print(f"  - Number of LSTM layers: {num_layers}")
        print(f"  - Dropout rate: {dropout}")
        print(f"  - Device: {device}")
    
    def forward(self, input_sequences: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the language model.
        
        Args:
            input_sequences: Input tensor of shape (batch_size, seq_length)
            hidden: Initial hidden state (h0, c0) for LSTM
            
        Returns:
            Tuple of (output_logits, (hidden_state, cell_state))
        """
        batch_size, seq_length = input_sequences.shape
        
        # Convert input indices to embeddings
        embeddings = self.embedding(input_sequences)  # (batch_size, seq_length, embedding_dim)
        
        # Apply dropout to embeddings
        embeddings = self.dropout(embeddings)
        
        # Pass through LSTM
        lstm_output, (hidden_state, cell_state) = self.lstm(embeddings, hidden)
        
        # Apply dropout to LSTM output
        lstm_output = self.dropout(lstm_output)
        
        # Pass through output layer to get logits
        output_logits = self.output_layer(lstm_output)  # (batch_size, seq_length, vocab_size)
        
        return output_logits, (hidden_state, cell_state)
    
    def init_hidden(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state for LSTM.
        
        Args:
            batch_size: Size of the batch
            
        Returns:
            Tuple of (hidden_state, cell_state) initialized to zeros
        """
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        return hidden_state, cell_state
    
    def calculate_loss(self, output_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the loss for the language model.
        
        Args:
            output_logits: Model output logits (batch_size, seq_length, vocab_size)
            targets: Target token indices (batch_size, seq_length)
            
        Returns:
            Loss value
        """
        # Reshape for loss calculation
        batch_size, seq_length, vocab_size = output_logits.shape
        output_logits = output_logits.view(-1, vocab_size)  # (batch_size * seq_length, vocab_size)
        targets = targets.view(-1)  # (batch_size * seq_length)
        
        # Calculate cross-entropy loss
        loss = nn.CrossEntropyLoss(ignore_index=self.vocab['<pad>'])(output_logits, targets)
        return loss
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            optimizer: Optimizer for updating weights
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (input_sequences, target_sequences) in enumerate(progress_bar):
            # Move data to device
            input_sequences = input_sequences.to(self.device)
            target_sequences = target_sequences.to(self.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output_logits, _ = self.forward(input_sequences)
            
            # Calculate loss
            loss = self.calculate_loss(output_logits, target_sequences)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            
            # Update weights
            optimizer.step()
            
            # Calculate accuracy
            with torch.no_grad():
                # Reshape for accuracy calculation
                batch_size, seq_length, vocab_size = output_logits.shape
                output_logits = output_logits.view(-1, vocab_size)
                targets = target_sequences.view(-1)
                
                # Get predicted tokens
                predicted = torch.argmax(output_logits, dim=1)
                
                # Calculate accuracy (ignore padding tokens)
                mask = targets != self.vocab['<pad>']
                correct = (predicted == targets) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate the model for one epoch.
        
        Args:
            val_loader: DataLoader for validation data
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.eval()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            
            for batch_idx, (input_sequences, target_sequences) in enumerate(progress_bar):
                # Move data to device
                input_sequences = input_sequences.to(self.device)
                target_sequences = target_sequences.to(self.device)
                
                # Forward pass
                output_logits, _ = self.forward(input_sequences)
                
                # Calculate loss
                loss = self.calculate_loss(output_logits, target_sequences)
                
                # Calculate accuracy
                batch_size, seq_length, vocab_size = output_logits.shape
                output_logits = output_logits.view(-1, vocab_size)
                targets = target_sequences.view(-1)
                
                # Get predicted tokens
                predicted = torch.argmax(output_logits, dim=1)
                
                # Calculate accuracy (ignore padding tokens)
                mask = targets != self.vocab['<pad>']
                correct = (predicted == targets) & mask
                total_correct += correct.sum().item()
                total_tokens += mask.sum().item()
                
                total_loss += loss.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
                })
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        
        return avg_loss, accuracy

# Simple vocabulary wrapper
class SimpleVocab:
    def __init__(self, stoi_dict):
        self.stoi = stoi_dict
        self.itos = {idx: token for token, idx in stoi_dict.items()}
        self._default_index = stoi_dict.get('<unk>', 0)
    
    def __getitem__(self, token):
        return self.stoi.get(token, self._default_index)
    
    def __len__(self):
        return len(self.stoi)
    
    def get_stoi(self):
        return self.stoi
    
    def get_itos(self):
        return self.itos
    
    def set_default_index(self, idx):
        self._default_index = idx

# Custom Dataset class
class IMDBDataset(torch.utils.data.Dataset):
    """Custom Dataset for IMDB language modeling."""
    def __init__(self, data_file: str, vocab: Any, seq_length: int):
        """
        Args:
            data_file: Path to the JSON file containing processed sequences
            vocab: Vocabulary object for token to index conversion
            seq_length: Length of sequences to use
        """
        with open(data_file, 'r') as f:
            self.sequences = json.load(f)
        self.vocab = vocab
        self.seq_length = seq_length

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_seq, target_seq = self.sequences[idx]
        
        # Convert input sequence to indices
        input_tokens = input_seq.split()
        input_indices = [self.vocab[token] for token in input_tokens]
        
        # Convert target sequence to indices
        target_tokens = target_seq.split()
        target_indices = [self.vocab[token] for token in target_tokens]
        
        return torch.tensor(input_indices), torch.tensor(target_indices)

def load_vocabulary(data_dir: str = './data/') -> SimpleVocab:
    """Load vocabulary from JSON file."""
    with open(f'{data_dir}/vocab.json', 'r') as f:
        vocab_stoi = json.load(f)
    
    return SimpleVocab(vocab_stoi)

def create_dataloaders(data_dir: str, vocab: SimpleVocab, batch_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """Create DataLoaders for train and validation sets."""
    train_dataset = IMDBDataset(f'{data_dir}/processed_lm_data/train.json', vocab, 50)
    val_dataset = IMDBDataset(f'{data_dir}/processed_lm_data/val.json', vocab, 50)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader

def train_model(model: LanguageModel, train_loader: DataLoader, val_loader: DataLoader, 
               epochs: int = 10, learning_rate: float = 0.001, 
               save_path: str = './model/language_model.pth') -> Dict[str, List[float]]:
    """
    Train the language model.
    
    Args:
        model: Language model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        save_path: Path to save the trained model
        
    Returns:
        Dictionary containing training history
    """
    print(f"\nStarting training for {epochs} epochs...")
    
    # Create directory for saving models
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=model.vocab['<pad>'])
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Training phase
        train_loss, train_accuracy = model.train_epoch(train_loader, optimizer, criterion)
        
        # Validation phase
        val_loss, val_accuracy = model.validate_epoch(val_loader, criterion)
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save model state dict and configuration
            model_config = {
                'embedding_dim': model.embedding_dim,
                'hidden_dim': model.hidden_dim,
                'num_layers': model.num_layers,
                'dropout': model.dropout.p,
                'vocab_size': len(model.vocab)
            }
            
            # Save model state dict
            torch.save(model.state_dict(), save_path)
            
            # Save model configuration separately
            config_path = save_path.replace('.pth', '_config.json')
            with open(config_path, 'w') as f:
                json.dump(model_config, f, indent=4)
            
            # Save training info
            training_info = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_val_loss,
                'model_config': model_config
            }
            info_path = save_path.replace('.pth', '_info.json')
            with open(info_path, 'w') as f:
                json.dump(training_info, f, indent=4)
            
            print(f"Model saved to {save_path}")
            print(f"Model config saved to {config_path}")
            print(f"Training info saved to {info_path}")
    
    # Plot training history
    plot_training_history(history)
    
    print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
    return history

def plot_training_history(history: Dict[str, List[float]]) -> None:
    """
    Plot training history.
    
    Args:
        history: Dictionary containing training history
    """
    # Check if we have enough data to plot
    if len(history['train_loss']) == 0:
        print("Warning: No training data available for plotting")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = list(range(len(history['train_loss'])))
    
    # Plot loss
    ax1.plot(epochs, history['train_loss'], 'bo-', label='Train Loss', linewidth=2, markersize=6)
    ax1.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss', linewidth=2, markersize=6)
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Add value annotations if only one data point
    if len(epochs) == 1:
        ax1.annotate(f'{history["train_loss"][0]:.4f}', (epochs[0], history['train_loss'][0]), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
        ax1.annotate(f'{history["val_loss"][0]:.4f}', (epochs[0], history['val_loss'][0]), 
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=10)
    
    # Plot accuracy
    ax2.plot(epochs, history['train_accuracy'], 'bo-', label='Train Accuracy', linewidth=2, markersize=6)
    ax2.plot(epochs, history['val_accuracy'], 'ro-', label='Validation Accuracy', linewidth=2, markersize=6)
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    # Add value annotations if only one data point
    if len(epochs) == 1:
        ax2.annotate(f'{history["train_accuracy"][0]:.4f}', (epochs[0], history['train_accuracy'][0]), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=10)
        ax2.annotate(f'{history["val_accuracy"][0]:.4f}', (epochs[0], history['val_accuracy'][0]), 
                    textcoords="offset points", xytext=(0,-15), ha='center', fontsize=10)
    
    plt.tight_layout()
    
    # Save with high quality settings
    save_path = './model/training_history.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Training history plot saved to {save_path}")
    print(f"Plotted {len(epochs)} epoch(s) of training data")

def main(data_dir: str = './data/', model_dir: str = './model/',
         embedding_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2,
         dropout: float = 0.3, learning_rate: float = 0.001, epochs: int = 10,
         batch_size: int = 32):
    """Main function to train the language model."""
    print("Step 5: Training Language Model")
    print("=" * 50)
    
    # Check if required files exist
    if not os.path.exists(f'{data_dir}/vocab.json'):
        print(f"Error: vocab.json not found in {data_dir}")
        print("Please run steps 1-4 first.")
        return
    if not os.path.exists(f'{data_dir}/processed_lm_data/train.json'):
        print(f"Error: train.json not found in {data_dir}/processed_lm_data/")
        print("Please run steps 1-4 first.")
        return
    if not os.path.exists(f'{data_dir}/processed_lm_data/val.json'):
        print(f"Error: val.json not found in {data_dir}/processed_lm_data/")
        print("Please run steps 1-4 first.")
        return
    
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load vocabulary
    vocab = load_vocabulary(data_dir)
    print(f"Loaded vocabulary with {len(vocab)} tokens")
    
    # Create DataLoaders
    train_loader, val_loader = create_dataloaders(data_dir, vocab, batch_size)
    print(f"Created DataLoaders:")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    model = LanguageModel(
        vocab=vocab,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        device=device
    )
    
    # Train the model
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        save_path=f'{model_dir}/language_model.pth'
    )
    
    print("\nStep 5 completed successfully!")
    print(f"Model saved to: {model_dir}/language_model.pth")
    print(f"Training history plot saved to: {model_dir}/training_history.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Language Model')
    parser.add_argument('--data-dir', type=str, default='./data/processed_lm_data/',
                       help='Directory containing processed data')
    parser.add_argument('--model-dir', type=str, default='./model/',
                       help='Directory to save model')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--hidden-dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    
    args = parser.parse_args()
    main(args.data_dir, args.model_dir, args.embedding_dim, args.hidden_dim,
         args.num_layers, args.dropout, args.learning_rate, args.epochs, args.batch_size) 