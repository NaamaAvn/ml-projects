import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
from torchtext.data.utils import get_tokenizer
import re
from typing import List, Dict, Any
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

def analyze_dataset(data: List, tokenizer, output_dir: str) -> Dict[str, Any]:
    """Analyze the dataset and generate visualizations.
    
    Args:
        data: List of (label, text) pairs
        tokenizer: Tokenizer function
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary containing analysis statistics
    """
    # Create output directory for plots
    plots_dir = os.path.join(output_dir, 'eda_plots')
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
    
    # 4. Calculate vocabulary coverage
    total_words = len(all_tokens)
    unique_words = len(set(all_tokens))
    
    coverage_stats = {
        'total_words': total_words,
        'unique_words': unique_words,
        'avg_sequence_length': np.mean(sequence_lengths),
        'median_sequence_length': np.median(sequence_lengths),
        'min_sequence_length': min(sequence_lengths),
        'max_sequence_length': max(sequence_lengths)
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

def save_analysis_stats(stats: Dict[str, Any], output_dir: str) -> None:
    """Save analysis statistics to file.
    
    Args:
        stats: Dictionary containing analysis statistics
        output_dir: Directory to save statistics
    """
    with open(f'{output_dir}/dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"Analysis statistics saved to {output_dir}/dataset_stats.json")

def main(data_dir: str = './data/processed_lm_data/', output_dir: str = './data/processed_lm_data/'):
    """Main function to analyze data and create EDA plots."""
    print("Step 2: Analyzing data and creating EDA plots")
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
    
    # Analyze training data (use training data for analysis)
    print("\nAnalyzing training data...")
    coverage_stats = analyze_dataset(train_data, tokenizer, output_dir)
    
    # Save analysis statistics
    save_analysis_stats(coverage_stats, output_dir)
    
    print("\nStep 2 completed successfully!")
    print(f"EDA plots saved to: {output_dir}/eda_plots/")
    print(f"Analysis statistics saved to: {output_dir}/dataset_stats.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze data and create EDA plots')
    parser.add_argument('--data-dir', type=str, default='./data/processed_lm_data/',
                       help='Directory containing data_splits.json')
    parser.add_argument('--output-dir', type=str, default='./data/processed_lm_data/',
                       help='Directory to save analysis results')
    
    args = parser.parse_args()
    main(args.data_dir, args.output_dir) 