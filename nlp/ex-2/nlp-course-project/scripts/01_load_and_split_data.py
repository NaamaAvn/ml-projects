import torch
from torchtext.datasets import IMDB
import os
import json
import random
from typing import List, Tuple
import argparse

def set_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)

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

def save_data_splits(train_data: List, val_data: List, test_data: List, 
                    output_dir: str = './data/processed_lm_data/') -> None:
    """Save data splits to JSON files.
    
    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        output_dir: Directory to save the splits
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save data splits
    data_splits = {
        'train_data': train_data,
        'val_data': val_data,
        'test_data': test_data
    }
    
    with open(f'{output_dir}/data_splits.json', 'w') as f:
        json.dump(data_splits, f)
    
    print(f"Data splits saved to {output_dir}/data_splits.json")
    print(f"Train data: {len(train_data):,} samples")
    print(f"Validation data: {len(val_data):,} samples")
    print(f"Test data: {len(test_data):,} samples")

def main(output_dir: str = './data/processed_lm_data/'):
    """Main function to load and split data."""
    print("Step 1: Loading and splitting IMDB dataset")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    set_seeds()
    
    # Load and split data
    train_data, val_data, test_data = load_and_split_data()
    
    # Save data splits
    save_data_splits(train_data, val_data, test_data, output_dir)
    
    print("\nStep 1 completed successfully!")
    print(f"Data splits saved to: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load and split IMDB dataset')
    parser.add_argument('--output-dir', type=str, default='./data/processed_lm_data/',
                       help='Directory to save data splits')
    
    args = parser.parse_args()
    main(args.output_dir) 