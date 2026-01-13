import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
import re
from typing import List, Tuple, Callable, Any
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

def clean_text(text: str) -> str:
    """Clean text by removing extra spaces and normalizing punctuation."""
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Add spaces around punctuation
    text = re.sub(r'([.,!?])', r' \1 ', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_data_splits(data_dir: str = './data/processed_lm_data/') -> tuple:
    """Load data splits from JSON file.
    
    Args:
        data_dir: Directory containing data_splits.json
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    with open(f'{data_dir}/data_splits.json', 'r') as f:
        data_splits = json.load(f)
    
    return data_splits['train_data'], data_splits['val_data'], data_splits['test_data']

def load_vocabulary(data_dir: str = './data/processed_lm_data/') -> SimpleVocab:
    """Load vocabulary from JSON file.
    
    Args:
        data_dir: Directory containing vocab.json
        
    Returns:
        SimpleVocab object
    """
    with open(f'{data_dir}/vocab.json', 'r') as f:
        vocab_stoi = json.load(f)
    
    return SimpleVocab(vocab_stoi)

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

def save_dataloader_configs(train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                          output_dir: str = './data/processed_lm_data/') -> None:
    """Save DataLoader configurations to file.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        output_dir: Directory to save configurations
    """
    dataloader_configs = {
        'train_batches': len(train_loader),
        'val_batches': len(val_loader),
        'test_batches': len(test_loader),
        'batch_size': train_loader.batch_size,
        'seq_length': 50
    }
    
    with open(f'{output_dir}/dataloader_configs.json', 'w') as f:
        json.dump(dataloader_configs, f)
    
    print(f"DataLoader configurations saved to {output_dir}/dataloader_configs.json")

def main(data_dir: str = './data/processed_lm_data/', output_dir: str = './data/processed_lm_data/',
         seq_length: int = 50, batch_size: int = 32, data_ratio: float = 1.0, test_ratio: float = 1.0):
    """Main function to create sequences and dataloaders."""
    print("Step 4: Creating sequences for language modeling and DataLoaders")
    print("=" * 60)
    
    # Check if required files exist
    required_files = ['data_splits.json', 'vocab.json']
    for file in required_files:
        if not os.path.exists(f'{data_dir}/{file}'):
            print(f"Error: {file} not found in {data_dir}")
            print("Please run steps 1-3 first.")
            return
    
    # Load data splits
    train_data, val_data, test_data = load_data_splits(data_dir)
    
    print(f"Loaded data splits:")
    print(f"Train data: {len(train_data):,} samples")
    print(f"Validation data: {len(val_data):,} samples")
    print(f"Test data: {len(test_data):,} samples")
    
    # Apply data subset ratios
    if data_ratio < 1.0:
        print(f"\nUsing {data_ratio*100:.1f}% of training/validation data...")
        train_subset_size = int(len(train_data) * data_ratio)
        val_subset_size = int(len(val_data) * data_ratio)
        train_data = train_data[:train_subset_size]
        val_data = val_data[:val_subset_size]
        print(f"  Training: {len(train_data):,} samples ({data_ratio*100:.1f}%)")
        print(f"  Validation: {len(val_data):,} samples ({data_ratio*100:.1f}%)")
    
    if test_ratio < 1.0:
        print(f"\nUsing {test_ratio*100:.1f}% of test data...")
        test_subset_size = int(len(test_data) * test_ratio)
        test_data = test_data[:test_subset_size]
        print(f"  Test: {len(test_data):,} samples ({test_ratio*100:.1f}%)")
    
    # Load vocabulary
    vocab = load_vocabulary(data_dir)
    print(f"\nLoaded vocabulary with {len(vocab)} tokens")
    
    # Initialize tokenizer
    tokenizer = get_tokenizer('basic_english')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process and save datasets
    print(f"\nProcessing datasets with sequence length {seq_length}...")
    train_size = process_dataset(iter(train_data), f'{output_dir}/train.json', seq_length, tokenizer)
    val_size = process_dataset(iter(val_data), f'{output_dir}/val.json', seq_length, tokenizer)
    test_size = process_dataset(iter(test_data), f'{output_dir}/test.json', seq_length, tokenizer)
    
    print(f"Processed {train_size:,} training sequences")
    print(f"Processed {val_size:,} validation sequences")
    print(f"Processed {test_size:,} test sequences")
    
    # Create DataLoaders
    print(f"\nCreating DataLoaders with batch size {batch_size}...")
    train_loader, val_loader, test_loader = create_dataloaders(
        f'{output_dir}/train.json',
        f'{output_dir}/val.json',
        f'{output_dir}/test.json',
        vocab,
        batch_size=batch_size,
        seq_length=seq_length
    )
    
    # Save DataLoader configurations
    save_dataloader_configs(train_loader, val_loader, test_loader, output_dir)
    
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
    
    # Print sample of processed data
    print("\nSample of processed training data (first sequence):")
    with open(f'{output_dir}/train.json', 'r') as f:
        train_sequences = json.load(f)
        if train_sequences:
            print("Input sequence:", train_sequences[0][0])
            print("Target sequence:", train_sequences[0][1])
    
    print("\nStep 4 completed successfully!")
    print(f"Processed sequences saved to: {output_dir}")
    print(f"DataLoaders created and ready for training")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create sequences for language modeling and DataLoaders')
    parser.add_argument('--data-dir', type=str, default='./data/processed_lm_data/',
                       help='Directory containing data_splits.json and vocab.json')
    parser.add_argument('--output-dir', type=str, default='./data/processed_lm_data/',
                       help='Directory to save processed sequences')
    parser.add_argument('--seq-length', type=int, default=50,
                       help='Length of sequences to create')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for DataLoaders')
    parser.add_argument('--data-ratio', type=float, default=1.0,
                       help='Ratio of training/validation data to use')
    parser.add_argument('--test-ratio', type=float, default=1.0,
                       help='Ratio of test data to use')
    
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.seq_length, args.batch_size, args.data_ratio, args.test_ratio) 