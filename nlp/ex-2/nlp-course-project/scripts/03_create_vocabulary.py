import json
import os
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import re
from typing import List, Any
import argparse

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

def yield_tokens(data_iter, tokenizer) -> List[str]:
    """Yield tokens from the dataset iterator."""
    for label, text in data_iter:
        # Convert text to string if it's not already
        if not isinstance(text, str):
            text = str(text)
        # Clean the text
        text = clean_text(text)
        yield tokenizer(text)

def build_vocabulary(train_data: List, tokenizer) -> Any:
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

def save_vocabulary(vocab: Any, output_dir: str = './data/processed_lm_data/') -> None:
    """Save vocabulary to JSON file.
    
    Args:
        vocab: Vocabulary object
        output_dir: Directory to save vocabulary
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save vocabulary
    with open(f'{output_dir}/vocab.json', 'w') as f:
        json.dump(vocab.get_stoi(), f)
    
    # Save tokenizer info
    tokenizer_info = {'type': 'basic_english'}
    with open(f'{output_dir}/tokenizer_info.json', 'w') as f:
        json.dump(tokenizer_info, f)
    
    print(f"Vocabulary saved to {output_dir}/vocab.json")
    print(f"Tokenizer info saved to {output_dir}/tokenizer_info.json")

def print_vocabulary_stats(vocab: Any) -> None:
    """Print vocabulary statistics.
    
    Args:
        vocab: Vocabulary object
    """
    print(f"\nVocabulary Statistics:")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Special tokens: <unk>={vocab['<unk>']}, <pad>={vocab['<pad>']}, <sos>={vocab['<sos>']}, <eos>={vocab['<eos>']}")
    
    print("\nMost common vocabulary items (first 20):")
    # Get the most common words (lowest IDs are typically the most common words)
    common_words = sorted(vocab.get_stoi().items(), key=lambda x: x[1])[:20]
    for word, idx in common_words:
        print(f"{word}: {idx}")

def main(data_dir: str = './data/processed_lm_data/', output_dir: str = './data/processed_lm_data/'):
    """Main function to create vocabulary."""
    print("Step 3: Creating vocabulary from training data")
    print("=" * 50)
    
    # Check if data splits exist
    if not os.path.exists(f'{data_dir}/data_splits.json'):
        print(f"Error: data_splits.json not found in {data_dir}")
        print("Please run step 1 (01_load_and_split_data.py) first.")
        return
    
    # Load data splits
    train_data, val_data, test_data = load_data_splits(data_dir)
    
    print(f"Loaded data splits:")
    print(f"Train data: {len(train_data):,} samples")
    print(f"Validation data: {len(val_data):,} samples")
    print(f"Test data: {len(test_data):,} samples")
    
    # Initialize tokenizer
    tokenizer = get_tokenizer('basic_english')
    
    # Build vocabulary from training data
    print("\nBuilding vocabulary from training data...")
    vocab = build_vocabulary(train_data, tokenizer)
    
    # Print vocabulary statistics
    print_vocabulary_stats(vocab)
    
    # Save vocabulary
    save_vocabulary(vocab, output_dir)
    
    print("\nStep 3 completed successfully!")
    print(f"Vocabulary saved to: {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create vocabulary from training data')
    parser.add_argument('--data-dir', type=str, default='./data/processed_lm_data/',
                       help='Directory containing data_splits.json')
    parser.add_argument('--output-dir', type=str, default='./data/processed_lm_data/',
                       help='Directory to save vocabulary')
    
    args = parser.parse_args()
    main(args.data_dir, args.output_dir) 