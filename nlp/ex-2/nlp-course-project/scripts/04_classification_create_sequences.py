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

class IMDBClassificationDataset(Dataset):
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

def clean_text(text: str) -> str:
    """Clean text by removing extra spaces and normalizing punctuation."""
    # Convert to string if not already
    if not isinstance(text, str):
        text = str(text)
    
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

def process_classification_dataset(data_iter, output_file: str, tokenizer: Callable) -> int:
    """Process dataset for classification and save to file.
    
    Args:
        data_iter: Iterator over dataset (label, text pairs)
        output_file: Path to save processed data
        tokenizer: Tokenizer function to use
        
    Returns:
        Number of samples processed
    """
    processed_data = []
    for item in data_iter:
        # IMDB data comes as [label, text] where label is 1 (negative) or 2 (positive)
        label, text = item
        
        # Convert IMDB labels (1,2) to binary classification labels (0,1)
        # 1 -> 0 (negative), 2 -> 1 (positive)
        binary_label = 1 if label == 2 else 0
        
        # Clean the text
        cleaned_text = clean_text(text)
        # Tokenize
        tokens = tokenizer(cleaned_text)
        # Join tokens back to string for storage
        processed_text = ' '.join(tokens)
        # Store as (text, label) pair
        processed_data.append((processed_text, binary_label))
    
    # Save processed data
    with open(output_file, 'w') as f:
        json.dump(processed_data, f)
    
    return len(processed_data)

def create_classification_dataloaders(train_file: str, val_file: str, test_file: str, 
                                    vocab: Any, batch_size: int = 32, max_length: int = 200) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for train, validation and test sets for classification."""
    train_dataset = IMDBClassificationDataset(train_file, vocab, max_length)
    val_dataset = IMDBClassificationDataset(val_file, vocab, max_length)
    test_dataset = IMDBClassificationDataset(test_file, vocab, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

def save_dataloader_configs(train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                          max_length: int, output_dir: str = './data/processed_classification_data/') -> None:
    """Save DataLoader configurations to file.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        max_length: Maximum sequence length used
        output_dir: Directory to save configurations
    """
    dataloader_configs = {
        'train_batches': len(train_loader),
        'val_batches': len(val_loader),
        'test_batches': len(test_loader),
        'batch_size': train_loader.batch_size,
        'max_length': max_length,
        'num_classes': 2,  # Binary classification (positive/negative)
        'vocab_size': len(train_loader.dataset.vocab)
    }
    
    with open(f'{output_dir}/dataloader_configs.json', 'w') as f:
        json.dump(dataloader_configs, f)
    
    print(f"DataLoader configurations saved to {output_dir}/dataloader_configs.json")

def main(data_dir: str = './data/processed_lm_data/', 
         output_dir: str = './data/processed_classification_data/',
         max_length: int = 200, batch_size: int = 32, 
         data_ratio: float = 1.0, test_ratio: float = 1.0):
    """Main function to create sequences and dataloaders for classification."""
    print("Step 4: Creating sequences for text classification and DataLoaders")
    print("=" * 70)
    
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
    print(f"\nProcessing datasets with max length {max_length}...")
    train_size = process_classification_dataset(iter(train_data), f'{output_dir}/train.json', tokenizer)
    val_size = process_classification_dataset(iter(val_data), f'{output_dir}/val.json', tokenizer)
    test_size = process_classification_dataset(iter(test_data), f'{output_dir}/test.json', tokenizer)
    
    print(f"Processed {train_size:,} training samples")
    print(f"Processed {val_size:,} validation samples")
    print(f"Processed {test_size:,} test samples")
    
    # Create DataLoaders
    print(f"\nCreating DataLoaders with batch size {batch_size}...")
    train_loader, val_loader, test_loader = create_classification_dataloaders(
        f'{output_dir}/train.json',
        f'{output_dir}/val.json',
        f'{output_dir}/test.json',
        vocab,
        batch_size=batch_size,
        max_length=max_length
    )
    
    # Save DataLoader configurations
    save_dataloader_configs(train_loader, val_loader, test_loader, max_length, output_dir)
    
    # Print DataLoader information
    print("\nDataLoader Information:")
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    print(f"Number of test batches: {len(test_loader)}")
    
    # Get a sample batch
    sample_batch = next(iter(train_loader))
    print("\nSample batch shapes:")
    print(f"Input shape: {sample_batch[0].shape}")
    print(f"Label shape: {sample_batch[1].shape}")
    
    # Print sample of processed data
    print("\nSample of processed training data (first sample):")
    with open(f'{output_dir}/train.json', 'r') as f:
        train_data = json.load(f)
        if train_data:
            print("Text:", train_data[0][0][:100] + "..." if len(train_data[0][0]) > 100 else train_data[0][0])
            print("Label:", train_data[0][1])
    
    # Print label distribution
    print("\nLabel distribution:")
    train_labels = [item[1] for item in train_data]
    val_labels = [item[1] for item in json.load(open(f'{output_dir}/val.json', 'r'))]
    test_labels = [item[1] for item in json.load(open(f'{output_dir}/test.json', 'r'))]
    
    print(f"Training - Negative: {train_labels.count(0):,}, Positive: {train_labels.count(1):,}")
    print(f"Validation - Negative: {val_labels.count(0):,}, Positive: {val_labels.count(1):,}")
    print(f"Test - Negative: {test_labels.count(0):,}, Positive: {test_labels.count(1):,}")
    
    print("\nStep 4 completed successfully!")
    print(f"Processed sequences saved to: {output_dir}")
    print(f"DataLoaders created and ready for classification training")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create sequences for text classification and DataLoaders')
    parser.add_argument('--data-dir', type=str, default='./data/',
                       help='Directory containing data_splits.json and vocab.json')
    parser.add_argument('--output-dir', type=str, default='./data/processed_classification_data/',
                       help='Directory to save processed sequences')
    parser.add_argument('--max-length', type=int, default=200,
                       help='Maximum length of sequences (will be padded/truncated)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for DataLoaders')
    parser.add_argument('--data-ratio', type=float, default=1.0,
                       help='Ratio of training/validation data to use')
    parser.add_argument('--test-ratio', type=float, default=1.0,
                       help='Ratio of test data to use')
    
    args = parser.parse_args()
    main(args.data_dir, args.output_dir, args.max_length, args.batch_size, args.data_ratio, args.test_ratio) 