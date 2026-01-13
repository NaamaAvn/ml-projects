import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset, DataLoader
import os
from typing import List, Tuple, Dict, Any, Callable
import json
import random
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time
from tqdm import tqdm
import argparse

# Create a simple vocabulary wrapper that mimics torchtext.vocab.Vocab
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

class IMDBDataset(Dataset):
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

def create_dataloaders(train_file: str, val_file: str, test_file: str, 
                      vocab: Any, batch_size: int = 32, seq_length: int = 50) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, validation and test sets."""
    train_dataset = IMDBDataset(train_file, vocab, seq_length)
    val_dataset = IMDBDataset(val_file, vocab, seq_length)
    test_dataset = IMDBDataset(test_file, vocab, seq_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)

def clean_text(text: str) -> str:
    """Clean text by removing extra spaces and normalizing punctuation."""
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Add spaces around punctuation
    text = re.sub(r'([.,!?])', r' \1 ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def yield_tokens(data_iter, tokenizer) -> List[str]:
    """Yield tokens from the dataset iterator."""
    for label, text in data_iter:
        # Convert text to string if it's not already
        if not isinstance(text, str):
            text = str(text)
        # Clean the text
        text = clean_text(text)
        yield tokenizer(text)

def create_lm_dataset(text: str, seq_length: int, tokenizer: Callable) -> List[Tuple[str, str]]:
    """Create input-target pairs for language modeling.
    
    Args:
        text: Input text string
        seq_length: Length of sequences to create
        tokenizer: Tokenizer function to use
        
    Returns:
        List of (input_sequence, target_sequence) pairs
    """
    # Convert text to string if it's not already
    if not isinstance(text, str):
        text = str(text)
    # Clean the text
    text = clean_text(text)
    tokens = tokenizer(text)
    sequences = []
    for i in range(0, len(tokens) - seq_length):
        input_seq = tokens[i:i + seq_length]
        target_seq = tokens[i + 1:i + seq_length + 1]
        sequences.append((' '.join(input_seq), ' '.join(target_seq)))
    return sequences

def load_and_split_data() -> Tuple[List, List, List]:
    """Load IMDB dataset and split into train, validation, and test sets.
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    train_iter = IMDB(split='train')
    test_iter = IMDB(split='test')
    
    # Convert iterators to lists for splitting
    train_data = list(train_iter)
    test_data = list(test_iter)
    
    # Shuffle training data
    random.shuffle(train_data)
    
    # Split training data into train and validation sets (90% train, 10% validation)
    split_idx = int(len(train_data) * 0.9)
    train_data, val_data = train_data[:split_idx], train_data[split_idx:]
    
    return train_data, val_data, test_data

def build_vocabulary(train_data: List, tokenizer: Callable) -> Any:
    """Build vocabulary from training data.
    
    Args:
        train_data: List of training examples
        tokenizer: Tokenizer function
        
    Returns:
        Vocabulary object
    """
    # Define special tokens
    specials = ['<unk>', '<pad>', '<sos>', '<eos>']
    
    vocab = build_vocab_from_iterator(
        yield_tokens(iter(train_data), tokenizer),
        specials=specials,
        min_freq=2  # Only include words that appear at least twice
    )
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def process_dataset(data_iter, output_file: str, seq_length: int, tokenizer: Callable) -> int:
    """Process dataset and save to file.
    
    Args:
        data_iter: Iterator over dataset
        output_file: Path to save processed data
        seq_length: Length of sequences to create
        tokenizer: Tokenizer function to use
        
    Returns:
        Number of sequences processed
    """
    sequences = []
    for label, text in data_iter:
        sequences.extend(create_lm_dataset(text, seq_length, tokenizer))
    
    # Save processed data
    with open(output_file, 'w') as f:
        json.dump(sequences, f)
    
    return len(sequences)

def save_dataset_statistics(stats: Dict[str, Any], output_file: str) -> None:
    """Save dataset statistics to file.
    
    Args:
        stats: Dictionary containing dataset statistics
        output_file: Path to save statistics
    """
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=4)

def analyze_dataset(data: List, vocab: Any, tokenizer: Callable, output_dir: str) -> dict:
    """Analyze the dataset and generate visualizations.
    
    Args:
        data: List of (label, text) pairs
        vocab: Vocabulary object
        tokenizer: Tokenizer function
        output_dir: Directory to save visualizations
    """
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Analyze sequence lengths
    sequence_lengths = []
    for _, text in data:
        tokens = tokenizer(clean_text(text))
        sequence_lengths.append(len(tokens))
    
    plt.figure(figsize=(10, 6))
    plt.hist(sequence_lengths, bins=50)
    plt.title('Distribution of Sequence Lengths')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plots_dir, 'sequence_lengths.png'))
    plt.close()
    
    # 2. Analyze label distribution
    # In IMDB dataset: 1 is positive (score >= 7), 0 is negative (score <= 4)
    labels = [int(label) for label, _ in data]  # Convert labels to int
    label_counts = Counter(labels)
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Negative', 'Positive'], [label_counts[1], label_counts[2]])
    plt.title('Distribution of Labels (Negative vs Positive Reviews)')
    plt.xlabel('Review Sentiment (Rating)')
    plt.ylabel('Count')
    
    # Add count labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom')
    
    # Add percentage labels
    total = sum(label_counts.values())
    for i, count in enumerate([label_counts[1], label_counts[2]]):
        percentage = (count/total) * 100
        plt.text(i, count/2, f'{percentage:.1f}%',
                ha='center', va='center',
                color='white' if count > total/10 else 'black')
    
    plt.savefig(os.path.join(plots_dir, 'label_distribution.png'))
    plt.close()
    
    # Print label distribution statistics
    print("\nLabel Distribution:")
    print(f"Negative reviews: {label_counts[1]:,} ({(label_counts[1]/total)*100:.1f}%)")
    print(f"Positive reviews: {label_counts[2]:,} ({(label_counts[2]/total)*100:.1f}%)")
    
    # 3. Analyze word frequency distribution
    all_tokens = []
    for _, text in data:
        tokens = tokenizer(clean_text(text))
        all_tokens.extend(tokens)
    
    word_freq = Counter(all_tokens)
    top_words = dict(word_freq.most_common(20))
    
    plt.figure(figsize=(12, 6))
    plt.bar(top_words.keys(), top_words.values())
    plt.title('Top 20 Most Frequent Words')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'word_frequency.png'))
    plt.close()
    
    # 4. Calculate and plot vocabulary coverage
    total_words = len(all_tokens)
    unique_words = len(set(all_tokens))
    vocab_coverage = len(vocab) / unique_words * 100
    
    coverage_stats = {
        'total_words': total_words,
        'unique_words': unique_words,
        'vocab_size': len(vocab),
        'vocab_coverage_percentage': vocab_coverage
    }
    
    # 5. Plot word length distribution
    word_lengths = [len(word) for word in all_tokens]
    plt.figure(figsize=(10, 6))
    plt.hist(word_lengths, bins=30)
    plt.title('Distribution of Word Lengths')
    plt.xlabel('Word Length')
    plt.ylabel('Count')
    plt.savefig(os.path.join(plots_dir, 'word_lengths.png'))
    plt.close()
    
    print("\nDataset Analysis Complete!")
    print(f"Visualizations saved in: {plots_dir}")
    print(f"Coverage statistics: {coverage_stats}")
    return coverage_stats

def perform_eda_and_preprocessing(output_dir: str = './data/processed_lm_data/') -> Tuple[List, List, List, Any]:
    """
    Perform EDA and preprocessing of the IMDB dataset.
    
    Args:
        output_dir: Directory to save processed data and visualizations
        
    Returns:
        Tuple containing:
        - train_data: List of training examples
        - val_data: List of validation examples
        - test_data: List of test examples
        - tokenizer: The tokenizer object
    """
    # Set random seeds
    set_seeds()
    
    # Initialize tokenizer
    tokenizer = get_tokenizer('basic_english')
    
    # Load and split data
    train_data, val_data, test_data = load_and_split_data()
    
    print(f"Train data: {len(train_data)}")
    print(f"Val data: {len(val_data)}")
    print(f"Test data: {len(test_data)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process and save datasets
    train_size = process_dataset(iter(train_data), f'{output_dir}/train.json', 50, tokenizer)
    val_size = process_dataset(iter(val_data), f'{output_dir}/val.json', 50, tokenizer)
    test_size = process_dataset(iter(test_data), f'{output_dir}/test.json', 50, tokenizer)
    
    # Build vocabulary for analysis
    vocab = build_vocabulary(train_data, tokenizer)
    
    # Run dataset analysis
    coverage_stats = analyze_dataset(train_data, vocab, tokenizer, output_dir)
    
    # Print statistics
    print(f"\nProcessed {train_size} training sequences")
    print(f"Processed {val_size} validation sequences")
    print(f"Processed {test_size} test sequences")
    
    # Print sample of processed data
    print("\nSample of processed training data (first sequence):")
    with open(f'{output_dir}/train.json', 'r') as f:
        train_sequences = json.load(f)
        if train_sequences:
            print("\nInput sequence:", train_sequences[0][0])
            print("Target sequence:", train_sequences[0][1])
    
    # Save dataset statistics
    stats = {
        'train_size': train_size,
        'val_size': val_size,
        'test_size': test_size,
        'sequence_length': 50
    }
    stats.update(coverage_stats)
    save_dataset_statistics(stats, f'{output_dir}/dataset_stats.json')
    
    return train_data, val_data, test_data, tokenizer

def build_vocab_and_dataloaders(train_data: List, tokenizer: Any, 
                              output_dir: str = './data/processed_lm_data/',
                              batch_size: int = 32) -> Tuple[Any, DataLoader, DataLoader, DataLoader]:
    """
    Build vocabulary and create DataLoaders for the dataset.
    
    Args:
        train_data: List of training examples
        tokenizer: Tokenizer object
        output_dir: Directory containing processed data
        batch_size: Batch size for DataLoaders
        
    Returns:
        Tuple containing:
        - vocab: Vocabulary object
        - train_loader: DataLoader for training data
        - val_loader: DataLoader for validation data
        - test_loader: DataLoader for test data
    """
    # Build vocabulary
    vocab = build_vocabulary(train_data, tokenizer)
    
    print(f"\nVocabulary size: {len(vocab)}")
    print("\nMost common vocabulary items (first 20):")
    # Get the most common words (lowest IDs are typically the most common words)
    common_words = sorted(vocab.get_stoi().items(), key=lambda x: x[1])[:20]
    for word, idx in common_words:
        print(f"{word}: {idx}")
    
    # Save vocabulary
    with open(f'{output_dir}/vocab.json', 'w') as f:
        json.dump(vocab.get_stoi(), f)
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        f'{output_dir}/train.json',
        f'{output_dir}/val.json',
        f'{output_dir}/test.json',
        vocab,
        batch_size=batch_size,
        seq_length=50
    )
    
    # Print DataLoader information
    print("\nDataLoader Information:")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Get a sample batch
    sample_batch = next(iter(train_loader))
    print("\nSample batch shapes:")
    print(f"Input shape: {sample_batch[0].shape}")
    print(f"Target shape: {sample_batch[1].shape}")
    
    return vocab, train_loader, val_loader, test_loader

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
    
    def generate_text(self, start_tokens: List[str], max_length: int = 50, temperature: float = 1.0) -> str:
        """
        Generate text using the trained model.
        
        Args:
            start_tokens: List of starting tokens
            max_length: Maximum length of generated text
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            Generated text as string
        """
        self.eval()
        generated_tokens = start_tokens.copy()
        
        with torch.no_grad():
            # Convert start tokens to indices
            input_indices = [self.vocab[token] for token in start_tokens]
            input_tensor = torch.tensor([input_indices]).to(self.device)
            
            # Initialize hidden state
            hidden = self.init_hidden(1)
            
            for _ in range(max_length):
                # Forward pass
                output_logits, hidden = self.forward(input_tensor, hidden)
                
                # Get the last output (for next word prediction)
                last_output = output_logits[:, -1, :]  # (1, vocab_size)
                
                # Apply temperature
                if temperature != 1.0:
                    last_output = last_output / temperature
                
                # Sample from the distribution
                probs = torch.softmax(last_output, dim=-1)
                next_token_idx = torch.multinomial(probs, 1).item()
                
                # Convert index back to token
                next_token = self.vocab.get_itos()[next_token_idx]
                
                # Stop if we generate an end token
                if next_token in ['<eos>', '<pad>']:
                    break
                
                generated_tokens.append(next_token)
                
                # Update input for next iteration
                input_tensor = torch.tensor([[next_token_idx]]).to(self.device)
        
        return ' '.join(generated_tokens)
    
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
    
    def train_model(self, train_loader: DataLoader, val_loader: DataLoader, 
               epochs: int = 10, learning_rate: float = 0.001, 
               save_path: str = './model/language_model.pth') -> Dict[str, List[float]]:
        """
        Train the language model.
        
        Args:
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
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss(ignore_index=self.vocab['<pad>'])
        
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
            train_loss, train_accuracy = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation phase
            val_loss, val_accuracy = self.validate_epoch(val_loader, criterion)
            
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
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'vocab': self.vocab,
                    'model_config': {
                        'embedding_dim': self.embedding_dim,
                        'hidden_dim': self.hidden_dim,
                        'num_layers': self.num_layers
                    }
                }, save_path)
                print(f"Model saved to {save_path}")
            
            # Generate sample text every few epochs
            if (epoch + 1) % 5 == 0:
                print("\nSample generated text:")
                sample_text = self.generate_text(['the', 'movie'], max_length=20)
                print(sample_text)
        
        # Plot training history
        self.plot_training_history(history)
        
        print(f"\nTraining completed! Best validation loss: {best_val_loss:.4f}")
        return history
    
    def plot_training_history(self, history: Dict[str, List[float]]) -> None:
        """
        Plot training history.
        
        Args:
            history: Dictionary containing training history
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracy
        ax2.plot(history['train_accuracy'], label='Train Accuracy')
        ax2.plot(history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('./model/training_history.png')
        plt.close()
        print("Training history plot saved to ./model/training_history.png")

def ensure_data_directory(output_dir: str = './data/processed_lm_data/') -> None:
    """
    Ensure the data directory exists.
    
    Args:
        output_dir: Directory to create if it doesn't exist
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Data directory ensured: {output_dir}")

def main(use_existing_data: bool = True, fast_training: bool = True, 
         data_subset_ratio: float = 0.1, test_subset_ratio: float = 0.1, max_epochs: int = 5):
    """
    Main function with options for faster training and using existing data.
    
    Args:
        use_existing_data: Whether to use existing processed data (skip steps 1-2)
        fast_training: Whether to use faster training parameters
        data_subset_ratio: Ratio of data to use for training/validation (0.1 = 10% of data)
        test_subset_ratio: Ratio of data to use for testing (0.1 = 10% of data)
        max_epochs: Maximum number of training epochs
    """
    print("Language Model Training")
    print("="*60)
    print(f"Use existing data: {use_existing_data}")
    print(f"Fast training: {fast_training}")
    print(f"Data subset ratio: {data_subset_ratio}")
    print(f"Test subset ratio: {test_subset_ratio}")
    print(f"Max epochs: {max_epochs}")
    print("="*60)
    
    # Ensure data directory exists
    ensure_data_directory()
    
    # Initialize variables
    train_data, val_data, test_data, tokenizer, vocab, train_loader, val_loader, test_loader = None, None, None, None, None, None, None, None
    
    # Step 1 & 2: Perform EDA and preprocessing (optional)
    if use_existing_data and check_intermediate_outputs():
        print("\nLoading existing processed data...")
        train_data, val_data, test_data, tokenizer, vocab, train_loader, val_loader, test_loader = load_intermediate_outputs()
        
        if train_data is None:
            print("Failed to load existing data. Running full preprocessing...")
            use_existing_data = False
    
    if not use_existing_data or train_loader is None:
        print("\nPerforming EDA and preprocessing...")
        train_data, val_data, test_data, tokenizer = perform_eda_and_preprocessing()
        
        print("\nBuilding vocabulary and creating DataLoaders...")
        vocab, train_loader, val_loader, test_loader = build_vocab_and_dataloaders(
            train_data, tokenizer
        )
        
        # Save intermediate outputs for future use
        save_intermediate_outputs(
            train_data, val_data, test_data, tokenizer, vocab,
            train_loader, val_loader, test_loader
        )
    
    # Apply data subset if requested
    if data_subset_ratio < 1.0:
        print(f"\nUsing {data_subset_ratio*100:.1f}% of training/validation data for faster training...")
        
        # Store original dataset sizes and batch counts
        original_train_size = len(train_loader.dataset)
        original_val_size = len(val_loader.dataset)
        original_train_batches = len(train_loader)
        original_val_batches = len(val_loader)
        
        # Create subset datasets
        train_subset_size = int(original_train_size * data_subset_ratio)
        val_subset_size = int(original_val_size * data_subset_ratio)
        
        # Create subset indices
        train_indices = torch.randperm(original_train_size)[:train_subset_size]
        val_indices = torch.randperm(original_val_size)[:val_subset_size]
        
        # Create subset datasets
        from torch.utils.data import Subset
        train_subset = Subset(train_loader.dataset, train_indices)
        val_subset = Subset(val_loader.dataset, val_indices)
        
        # Create new DataLoaders with subsets
        train_loader = DataLoader(train_subset, batch_size=train_loader.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=val_loader.batch_size)
        
        print(f"Data reduction:")
        print(f"  Training: {original_train_size:,} → {train_subset_size:,} samples ({data_subset_ratio*100:.1f}%)")
        print(f"  Validation: {original_val_size:,} → {val_subset_size:,} samples ({data_subset_ratio*100:.1f}%)")
        print(f"  Batches: {original_train_batches} → {len(train_loader)} training, {original_val_batches} → {len(val_loader)} validation")
    
    # Apply test subset if requested
    if test_loader is not None and test_subset_ratio < 1.0:
        print(f"\nUsing {test_subset_ratio*100:.1f}% of test data for faster evaluation...")
        
        # Store original test dataset size and batch count
        original_test_size = len(test_loader.dataset)
        original_test_batches = len(test_loader)
        
        # Create test subset
        test_subset_size = int(original_test_size * test_subset_ratio)
        test_indices = torch.randperm(original_test_size)[:test_subset_size]
        
        # Create test subset dataset
        from torch.utils.data import Subset
        test_subset = Subset(test_loader.dataset, test_indices)
        
        # Create new test DataLoader with subset
        test_loader = DataLoader(test_subset, batch_size=test_loader.batch_size)
        
        print(f"Test data reduction:")
        print(f"  Test: {original_test_size:,} → {test_subset_size:,} samples ({test_subset_ratio*100:.1f}%)")
        print(f"  Test batches: {original_test_batches} → {len(test_loader)}")

    # Step 3: Define and train the language model
    print("\nDefining and training the language model...")
    
    # Detect device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Choose model parameters based on fast_training flag
    if fast_training:
        print("Using fast training parameters...")
        model_params = {
            'embedding_dim': 64,    # Reduced from 128
            'hidden_dim': 128,      # Reduced from 256
            'num_layers': 1,        # Reduced from 2
            'dropout': 0.2,         # Reduced from 0.3
            'learning_rate': 0.01,  # Increased from 0.001
            'batch_size': 64        # Increased from 32
        }
    else:
        print("Using full training parameters...")
        model_params = {
            'embedding_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32
        }
    
    # Initialize model
    model = LanguageModel(
        vocab=vocab,
        embedding_dim=model_params['embedding_dim'],
        hidden_dim=model_params['hidden_dim'],
        num_layers=model_params['num_layers'],
        dropout=model_params['dropout'],
        device=device
    )
    
    # Train the model
    history = model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=max_epochs,
        learning_rate=model_params['learning_rate'],
        save_path='./model/language_model.pth'
    )
    
    # Step 4: Test the trained model (optional, only if we have test data)
    if test_loader is not None:
        print("\nTesting the trained model...")
        test_trained_model(model, test_loader, device)
    
    print("\nTraining completed!")
    print(f"Model saved to: ./model/language_model.pth")
    print(f"Training history plot saved to: ./model/training_history.png")

def train_fast_demo():
    """
    Quick demo function for fast training on a small subset.
    """
    print("Running fast training demo...")
    main(
        use_existing_data=True,
        fast_training=True,
        data_subset_ratio=0.05,  # Use only 5% of data
        test_subset_ratio=0.05,  # Use only 5% of test data
        max_epochs=3
    )

def train_full_model():
    """
    Full training function for production use.
    """
    print("Running full model training...")
    main(
        use_existing_data=True,
        fast_training=False,
        data_subset_ratio=1.0,  # Use all data
        test_subset_ratio=1.0,  # Use all test data
        max_epochs=15
    )

def test_trained_model(model: LanguageModel, test_loader: DataLoader, device: str) -> None:
    """
    Test the trained language model on test data.
    
    Args:
        model: Trained language model
        test_loader: DataLoader for test data
        device: Device to run the model on
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    print("Evaluating on test set...")
    
    with torch.no_grad():
        for batch_idx, (input_sequences, target_sequences) in enumerate(tqdm(test_loader, desc="Testing")):
            # Move data to device
            input_sequences = input_sequences.to(device)
            target_sequences = target_sequences.to(device)
            
            # Forward pass
            output_logits, _ = model.forward(input_sequences)
            
            # Calculate loss
            loss = model.calculate_loss(output_logits, target_sequences)
            
            # Calculate accuracy
            batch_size, seq_length, vocab_size = output_logits.shape
            output_logits = output_logits.view(-1, vocab_size)
            targets = target_sequences.view(-1)
            
            # Get predicted tokens
            predicted = torch.argmax(output_logits, dim=1)
            
            # Calculate accuracy (ignore padding tokens)
            mask = targets != model.vocab['<pad>']
            correct = (predicted == targets) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
    
    print(f"\nTest Results:")
    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Generate some sample text
    print("\nSample generated texts:")
    sample_prompts = [
        ['the', 'movie', 'was'],
        ['i', 'really', 'liked'],
        ['this', 'film', 'is'],
        ['the', 'acting', 'was'],
        ['the', 'story', 'is']
    ]
    
    for prompt in sample_prompts:
        generated_text = model.generate_text(prompt, max_length=30, temperature=0.8)
        print(f"Prompt: {' '.join(prompt)}")
        print(f"Generated: {generated_text}")
        print("-" * 50)

def load_and_test_model(model_path: str = './model/language_model.pth') -> LanguageModel:
    """
    Load a trained model and test it.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded language model
    """
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None
    
    # Load the model checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get model configuration
    model_config = checkpoint['model_config']
    vocab = checkpoint['vocab']
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LanguageModel(
        vocab=vocab,
        embedding_dim=model_config['embedding_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        device=device
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")
    print(f"Training completed at epoch {checkpoint['epoch'] + 1}")
    print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
    
    return model

def save_intermediate_outputs(train_data: List, val_data: List, test_data: List, 
                            tokenizer: Any, vocab: Any, train_loader: DataLoader, 
                            val_loader: DataLoader, test_loader: DataLoader,
                            output_dir: str = './data/processed_lm_data/') -> None:
    """
    Save intermediate outputs from steps 1 and 2 for later use.
    
    Args:
        train_data: Training data
        val_data: Validation data  
        test_data: Test data
        tokenizer: Tokenizer object
        vocab: Vocabulary object
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        output_dir: Directory to save outputs
    """
    print("Saving intermediate outputs...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data splits
    data_splits = {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data
    }
    
    with open(f'{output_dir}/data_splits.json', 'w') as f:
        json.dump(data_splits, f)
    
    # Save tokenizer info (basic_english tokenizer is standard)
    tokenizer_info = {'type': 'basic_english'}
    with open(f'{output_dir}/tokenizer_info.json', 'w') as f:
        json.dump(tokenizer_info, f)
    
    # Save vocabulary
    with open(f'{output_dir}/vocab.json', 'w') as f:
        json.dump(vocab.get_stoi(), f)
    
    # Save DataLoader info (we can't pickle DataLoaders directly, so save their configs)
    dataloader_configs = {
        'train_batches': len(train_loader),
        'val_batches': len(val_loader),
        'test_batches': len(test_loader),
        'batch_size': train_loader.batch_size,
        'seq_length': 50
    }
    
    with open(f'{output_dir}/dataloader_configs.json', 'w') as f:
        json.dump(dataloader_configs, f)
    
    print(f"Intermediate outputs saved to {output_dir}")

def load_intermediate_outputs(output_dir: str = './data/processed_lm_data/') -> Tuple[List, List, List, Any, Any, DataLoader, DataLoader, DataLoader]:
    """
    Load intermediate outputs from steps 1 and 2.
    
    Args:
        output_dir: Directory containing saved outputs
        
    Returns:
        Tuple of (train_data, val_data, test_data, tokenizer, vocab, train_loader, val_loader, test_loader)
    """
    print("Loading intermediate outputs...")
    
    # Check if all required files exist
    required_files = [
        'data_splits.json',
        'tokenizer_info.json', 
        'vocab.json',
        'dataloader_configs.json',
        'train.json',
        'val.json',
        'test.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(f'{output_dir}/{file}'):
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing required files: {missing_files}")
        print("Please run steps 1 and 2 first or ensure all files are present.")
        return None, None, None, None, None, None, None, None
    
    # Load data splits
    with open(f'{output_dir}/data_splits.json', 'r') as f:
        data_splits = json.load(f)
        train_data = data_splits['train_data']
        val_data = data_splits['val_data']
        test_data = data_splits['test_data']
    
    # Load tokenizer
    with open(f'{output_dir}/tokenizer_info.json', 'r') as f:
        tokenizer_info = json.load(f)
    
    tokenizer = get_tokenizer(tokenizer_info['type'])
    
    # Load vocabulary
    with open(f'{output_dir}/vocab.json', 'r') as f:
        vocab_stoi = json.load(f)
    
    # Create vocabulary object
    vocab = SimpleVocab(vocab_stoi)
    
    # Verify vocabulary is working correctly
    try:
        # Test that we can access special tokens
        unk_idx = vocab['<unk>']
        pad_idx = vocab['<pad>']
        print(f"Vocabulary loaded successfully. Size: {len(vocab)}")
        print(f"Special tokens: <unk>={unk_idx}, <pad>={pad_idx}")
    except Exception as e:
        print(f"Error with vocabulary reconstruction: {e}")
        print("Falling back to full preprocessing...")
        return None, None, None, None, None, None, None, None
    
    # Load DataLoader configs and recreate DataLoaders
    with open(f'{output_dir}/dataloader_configs.json', 'r') as f:
        dataloader_configs = json.load(f)
    
    # Recreate DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        f'{output_dir}/train.json',
        f'{output_dir}/val.json', 
        f'{output_dir}/test.json',
        vocab,
        batch_size=dataloader_configs['batch_size'],
        seq_length=dataloader_configs['seq_length']
    )
    
    print("Intermediate outputs loaded successfully!")
    return train_data, val_data, test_data, tokenizer, vocab, train_loader, val_loader, test_loader

def check_intermediate_outputs(output_dir: str = './data/processed_lm_data/') -> bool:
    """
    Check if intermediate outputs from steps 1 and 2 exist.
    
    Args:
        output_dir: Directory to check
        
    Returns:
        True if all required files exist, False otherwise
    """
    required_files = [
        'data_splits.json',
        'tokenizer_info.json',
        'vocab.json', 
        'dataloader_configs.json',
        'train.json',
        'val.json',
        'test.json'
    ]
    
    for file in required_files:
        if not os.path.exists(f'{output_dir}/{file}'):
            return False
    
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Language Model on IMDB Dataset')
    parser.add_argument('--mode', type=str, default='fast', 
                       choices=['fast', 'full', 'custom'],
                       help='Training mode: fast (quick demo), full (production), or custom')
    parser.add_argument('--use-existing', action='store_true', default=True,
                       help='Use existing processed data if available')
    parser.add_argument('--data-ratio', type=float, default=0.1,
                       help='Ratio of data to use for training (0.1 = 10%%)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Ratio of data to use for testing (0.1 = 10%%)')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--fast-params', action='store_true', default=True,
                       help='Use fast training parameters (smaller model)')
    
    args = parser.parse_args()
    
    if args.mode == 'fast':
        print("Running fast training demo...")
        train_fast_demo()
    elif args.mode == 'full':
        print("Running full model training...")
        train_full_model()
    elif args.mode == 'custom':
        print("Running custom training...")
        main(
            use_existing_data=args.use_existing,
            fast_training=args.fast_params,
            data_subset_ratio=args.data_ratio,
            test_subset_ratio=args.test_ratio,
            max_epochs=args.epochs
        )
    else:
        # Default: fast training
        print("Running fast training demo...")
        train_fast_demo() 