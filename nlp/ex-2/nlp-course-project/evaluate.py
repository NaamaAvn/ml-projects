# Ensure NLTK resources are available
try:
    import nltk
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from collections import Counter
import math
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import warnings
warnings.filterwarnings('ignore')

# Import the LanguageModel class from train.py
from train import LanguageModel, SimpleVocab, IMDBDataset, smart_detokenize

class LanguageModelEvaluator:
    """
    Comprehensive evaluator for language models with multiple metrics.
    
    Supports:
    - Perplexity (standard LM metric)
    - BLEU score (for text generation quality)
    - ROUGE score (for text generation quality)
    - METEOR score (for text generation quality)
    - Token-level accuracy
    """
    
    def __init__(self, model: LanguageModel, vocab: SimpleVocab, device: str = 'cpu'):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained language model
            vocab: Vocabulary object
            device: Device to run evaluation on
        """
        self.model = model
        self.vocab = vocab
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Initialize ROUGE scorer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        print(f"Language Model Evaluator initialized on device: {device}")
    
    def calculate_perplexity(self, test_loader: DataLoader) -> Tuple[float, float]:
        """
        Calculate perplexity on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Tuple of (Perplexity, Average Loss)
        """
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for input_sequences, target_sequences in tqdm(test_loader, desc="Calculating Perplexity & Loss"):
                input_sequences = input_sequences.to(self.device)
                target_sequences = target_sequences.to(self.device)
                
                # Forward pass
                output_logits, _ = self.model.forward(input_sequences)
                
                # Calculate loss
                loss = self.model.calculate_loss(output_logits, target_sequences)
                
                # Count non-padding tokens
                batch_size, seq_length, vocab_size = output_logits.shape
                output_logits = output_logits.view(-1, vocab_size)
                targets = target_sequences.view(-1)
                
                # Count non-padding tokens
                mask = targets != self.vocab['<pad>']
                num_tokens = mask.sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Calculate average loss and perplexity
        avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
        perplexity = math.exp(avg_loss)
        
        return perplexity, avg_loss
    
    def calculate_accuracy(self, test_loader: DataLoader) -> float:
        """
        Calculate token-level accuracy on test data.
        
        Args:
            test_loader: DataLoader for test data
            
        Returns:
            Accuracy value
        """
        self.model.eval()
        total_correct = 0
        total_tokens = 0
        
        with torch.no_grad():
            for input_sequences, target_sequences in tqdm(test_loader, desc="Calculating Accuracy"):
                input_sequences = input_sequences.to(self.device)
                target_sequences = target_sequences.to(self.device)
                
                # Forward pass
                output_logits, _ = self.model.forward(input_sequences)
                
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
        
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        return accuracy
    
    def generate_text_samples(self, num_samples: int = 100, max_length: int = 50, 
                            temperature: float = 1.0) -> List[str]:
        """
        Generate text samples for evaluation.
        
        Args:
            num_samples: Number of samples to generate
            max_length: Maximum length of each sample
            temperature: Temperature for sampling
            
        Returns:
            List of generated text samples
        """
        self.model.eval()
        generated_samples = []
        
        # Common starting tokens for IMDB reviews
        start_tokens_list = [
            ['the', 'movie', 'was'],
            ['this', 'film', 'is'],
            ['i', 'really', 'liked'],
            ['the', 'acting', 'was'],
            ['this', 'is', 'a'],
            ['the', 'story', 'is'],
            ['i', 'thought', 'this'],
            ['the', 'director', 'did'],
            ['this', 'movie', 'has'],
            ['the', 'plot', 'was']
        ]
        stoi = self.vocab.get_stoi()
        unk_idx = self.vocab['<unk>']
        
        with torch.no_grad():
            for i in tqdm(range(num_samples), desc="Generating Text Samples"):
                # Choose random starting tokens
                start_tokens = start_tokens_list[i % len(start_tokens_list)]
                
                # Convert to indices
                start_indices = [self.vocab[token] if token in stoi else unk_idx for token in start_tokens]
                
                # Generate text
                generated_text = self._generate_text_from_tokens(
                    start_indices, max_length, temperature
                )
                
                generated_samples.append(generated_text)
        
        return generated_samples
    
    def _generate_text_from_tokens(self, start_tokens: List[int], max_length: int, 
                                  temperature: float) -> str:
        """
        Generate text from starting tokens.
        
        Args:
            start_tokens: List of starting token indices
            max_length: Maximum length to generate
            temperature: Temperature for sampling
            
        Returns:
            Generated text string
        """
        self.model.eval()
        
        # Initialize with start tokens
        current_tokens = torch.tensor([start_tokens], dtype=torch.long).to(self.device)
        generated_tokens = start_tokens.copy()
        itos = self.vocab.get_itos()
        unk_idx = self.vocab['<unk>']
        pad_idx = self.vocab['<pad>']
        eos_idx = self.vocab['<eos>'] if '<eos>' in self.vocab.get_stoi() else -1
        
        # Create mask to exclude special tokens from sampling
        special_tokens = {unk_idx, pad_idx}
        if eos_idx != -1:
            special_tokens.add(eos_idx)
        
        with torch.no_grad():
            for _ in range(max_length - len(start_tokens)):
                # Forward pass
                output_logits, _ = self.model.forward(current_tokens)
                
                # Get logits for the last token
                last_logits = output_logits[0, -1, :]
                
                # Mask out special tokens by setting their logits to -inf
                for special_idx in special_tokens:
                    last_logits[special_idx] = float('-inf')
                
                # Apply temperature
                if temperature != 1.0:
                    last_logits = last_logits / temperature
                
                # Sample from distribution
                probs = torch.softmax(last_logits, dim=-1)
                next_token = torch.multinomial(probs, 1).item()
                
                # Add to generated tokens
                generated_tokens.append(next_token)
                
                # Update current tokens for next iteration
                current_tokens = torch.tensor([generated_tokens], dtype=torch.long).to(self.device)
        
        # Convert back to text
        generated_tokens_text = [itos.get(idx, '<unk>') for idx in generated_tokens]
        return smart_detokenize(generated_tokens_text)
    
    def calculate_bleu_scores(self, generated_samples: List[str], 
                            reference_samples: List[str]) -> Dict[str, float]:
        """
        Calculate BLEU scores for generated text.
        
        Args:
            generated_samples: List of generated text samples
            reference_samples: List of reference text samples
            
        Returns:
            Dictionary with BLEU scores
        """
        # Tokenize texts using simple split (avoiding NLTK tokenizer issues)
        generated_tokens = [sample.lower().split() for sample in generated_samples]
        reference_tokens = [sample.lower().split() for sample in reference_samples]
        
        # Calculate BLEU scores
        smoothing = SmoothingFunction().method1
        
        bleu_1_scores = []
        bleu_2_scores = []
        bleu_3_scores = []
        bleu_4_scores = []
        
        for gen_tokens, ref_tokens in zip(generated_tokens, reference_tokens):
            # BLEU-1
            bleu_1 = sentence_bleu([ref_tokens], gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu_1_scores.append(bleu_1)
            
            # BLEU-2
            bleu_2 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu_2_scores.append(bleu_2)
            
            # BLEU-3
            bleu_3 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu_3_scores.append(bleu_3)
            
            # BLEU-4
            bleu_4 = sentence_bleu([ref_tokens], gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            bleu_4_scores.append(bleu_4)
        
        return {
            'bleu_1': np.mean(bleu_1_scores),
            'bleu_2': np.mean(bleu_2_scores),
            'bleu_3': np.mean(bleu_3_scores),
            'bleu_4': np.mean(bleu_4_scores)
        }
    
    def calculate_rouge_scores(self, generated_samples: List[str], 
                             reference_samples: List[str]) -> Dict[str, float]:
        """
        Calculate ROUGE scores for generated text.
        
        Args:
            generated_samples: List of generated text samples
            reference_samples: List of reference text samples
            
        Returns:
            Dictionary with ROUGE scores
        """
        rouge_scores = {
            'rouge1_precision': [], 'rouge1_recall': [], 'rouge1_fmeasure': [],
            'rouge2_precision': [], 'rouge2_recall': [], 'rouge2_fmeasure': [],
            'rougeL_precision': [], 'rougeL_recall': [], 'rougeL_fmeasure': []
        }
        
        for gen_text, ref_text in zip(generated_samples, reference_samples):
            scores = self.rouge_scorer.score(ref_text, gen_text)
            
            # ROUGE-1
            rouge_scores['rouge1_precision'].append(scores['rouge1'].precision)
            rouge_scores['rouge1_recall'].append(scores['rouge1'].recall)
            rouge_scores['rouge1_fmeasure'].append(scores['rouge1'].fmeasure)
            
            # ROUGE-2
            rouge_scores['rouge2_precision'].append(scores['rouge2'].precision)
            rouge_scores['rouge2_recall'].append(scores['rouge2'].recall)
            rouge_scores['rouge2_fmeasure'].append(scores['rouge2'].fmeasure)
            
            # ROUGE-L
            rouge_scores['rougeL_precision'].append(scores['rougeL'].precision)
            rouge_scores['rougeL_recall'].append(scores['rougeL'].recall)
            rouge_scores['rougeL_fmeasure'].append(scores['rougeL'].fmeasure)
        
        # Calculate averages
        avg_scores = {}
        for metric, values in rouge_scores.items():
            avg_scores[metric] = np.mean(values)
        
        return avg_scores
    
    def get_reference_samples(self, test_loader: DataLoader, num_samples: int = 100) -> List[str]:
        """
        Extract reference samples from test data.
        
        Args:
            test_loader: DataLoader for test data
            num_samples: Number of samples to extract
            
        Returns:
            List of reference text samples
        """
        reference_samples = []
        itos = self.vocab.get_itos()
        
        for batch_idx, (input_sequences, target_sequences) in enumerate(test_loader):
            if len(reference_samples) >= num_samples:
                break
                
            batch_size = input_sequences.size(0)
            for i in range(min(batch_size, num_samples - len(reference_samples))):
                # Convert target sequence to text
                target_tokens = target_sequences[i].tolist()
                # Remove padding tokens
                target_tokens = [token for token in target_tokens if token != self.vocab['<pad>']]
                # Convert to text
                text = ' '.join([itos.get(idx, '<unk>') for idx in target_tokens])
                reference_samples.append(text)
        
        return reference_samples
    
    def evaluate_model(self, test_loader: DataLoader, num_generation_samples: int = 100) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            test_loader: DataLoader for test data
            num_generation_samples: Number of samples to generate for text quality metrics
            
        Returns:
            Dictionary with all evaluation metrics
        """
        print("Starting comprehensive model evaluation...")
        
        # Calculate perplexity and loss
        print("\n1. Calculating Perplexity and Loss...")
        perplexity, test_loss = self.calculate_perplexity(test_loader)
        
        # Calculate accuracy
        print("\n2. Calculating Token-level Accuracy...")
        accuracy = self.calculate_accuracy(test_loader)
        
        # Generate text samples
        print("\n3. Generating Text Samples...")
        generated_samples = self.generate_text_samples(num_generation_samples)
        
        # Get reference samples
        print("\n4. Extracting Reference Samples...")
        reference_samples = self.get_reference_samples(test_loader, num_generation_samples)
        
        # Calculate BLEU scores
        print("\n5. Calculating BLEU Scores...")
        bleu_scores = self.calculate_bleu_scores(generated_samples, reference_samples)
        
        # Calculate ROUGE scores
        print("\n6. Calculating ROUGE Scores...")
        rouge_scores = self.calculate_rouge_scores(generated_samples, reference_samples)
        
        # Compile results
        results = {
            'perplexity': perplexity,
            'test_loss': test_loss,
            'accuracy': accuracy,
            'bleu_scores': bleu_scores,
            'rouge_scores': rouge_scores,
            'generated_samples': generated_samples[:10],  # Save first 10 samples
            'reference_samples': reference_samples[:10]   # Save first 10 samples
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """
        Print evaluation results in a formatted way.
        
        Args:
            results: Evaluation results dictionary
        """
        print("\n" + "="*60)
        print("LANGUAGE MODEL EVALUATION RESULTS")
        print("="*60)
        
        # Basic metrics
        print(f"\n📊 CORE METRICS:")
        print(f"   Test Loss: {results['test_loss']:.4f}")
        print(f"   Perplexity: {results['perplexity']:.4f}")
        print(f"   Token-level Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        
        # BLEU scores
        print(f"\n🎯 BLEU SCORES:")
        bleu = results['bleu_scores']
        print(f"   BLEU-1: {bleu['bleu_1']:.4f}")
        print(f"   BLEU-2: {bleu['bleu_2']:.4f}")
        print(f"   BLEU-3: {bleu['bleu_3']:.4f}")
        print(f"   BLEU-4: {bleu['bleu_4']:.4f}")
        
        # ROUGE scores
        print(f"\n🔍 ROUGE SCORES:")
        rouge = results['rouge_scores']
        print(f"   ROUGE-1 Precision: {rouge['rouge1_precision']:.4f}")
        print(f"   ROUGE-1 Recall: {rouge['rouge1_recall']:.4f}")
        print(f"   ROUGE-1 F1: {rouge['rouge1_fmeasure']:.4f}")
        print(f"   ROUGE-2 Precision: {rouge['rouge2_precision']:.4f}")
        print(f"   ROUGE-2 Recall: {rouge['rouge2_recall']:.4f}")
        print(f"   ROUGE-2 F1: {rouge['rouge2_fmeasure']:.4f}")
        print(f"   ROUGE-L Precision: {rouge['rougeL_precision']:.4f}")
        print(f"   ROUGE-L Recall: {rouge['rougeL_recall']:.4f}")
        print(f"   ROUGE-L F1: {rouge['rougeL_fmeasure']:.4f}")
        
        print("="*60)
    
    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """
        Save evaluation results to file.
        
        Args:
            results: Evaluation results dictionary
            output_path: Path to save results
        """
        # Create a copy without the sample texts for JSON serialization
        results_copy = results.copy()
        
        # Handle different result types
        if 'generated_samples' in results_copy:
            results_copy['generated_samples'] = results_copy['generated_samples'][:5]  # Keep only first 5
        if 'reference_samples' in results_copy:
            results_copy['reference_samples'] = results_copy['reference_samples'][:5]  # Keep only first 5
        
        with open(output_path, 'w') as f:
            json.dump(results_copy, f, indent=2)
        
        print(f"Results saved to: {output_path}")


def load_model_and_vocab(model_path: str, vocab_path: str, device: str = 'cpu') -> Tuple[LanguageModel, SimpleVocab]:
    """
    Load trained model and vocabulary.
    
    Args:
        model_path: Path to trained model
        vocab_path: Path to vocabulary file
        device: Device to load model on
        
    Returns:
        Tuple of (model, vocab)
    """
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    vocab = SimpleVocab(vocab_data)
    
    # Load model configuration
    config_path = model_path.replace('.pth', '_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model with saved configuration
        model = LanguageModel(
            vocab=vocab,
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            device=device
        )
    else:
        # Fallback to default configuration
        print("Warning: Model config file not found, using default parameters")
        model = LanguageModel(vocab, device=device)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    return model, vocab


def create_test_dataloader(data_dir: str, vocab: SimpleVocab, batch_size: int = 32) -> DataLoader:
    """
    Create test data loader.
    
    Args:
        data_dir: Directory containing test data
        vocab: Vocabulary object
        batch_size: Batch size for data loader
        
    Returns:
        DataLoader for test data
    """
    test_file = os.path.join(data_dir, 'processed_lm_data', 'test.json')
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test data file not found: {test_file}")
    
    # Create dataset (pass file path, not loaded data)
    test_dataset = IMDBDataset(test_file, vocab, seq_length=50)  # Assuming seq_length=50
    
    # Create data loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader


def main(results_dir: str = './results/'):
    """
    Main evaluation function.
    
    Args:
        results_dir: Directory to save evaluation results
    """
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    parser = argparse.ArgumentParser(description='Evaluate Language Model')
    parser.add_argument('--model-path', type=str, default='./model/language_model.pth',
                       help='Path to trained model')
    parser.add_argument('--vocab-path', type=str, default='./data/vocab.json',
                       help='Path to vocabulary file')
    parser.add_argument('--data-dir', type=str, default='./data/',
                       help='Directory containing test data')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run evaluation on (cpu/cuda)')
    parser.add_argument('--results-dir', type=str, default='./results/',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    
    try:
        # Load model and vocabulary
        print("Loading model and vocabulary...")
        model, vocab = load_model_and_vocab(args.model_path, args.vocab_path, args.device)
        
        # Create evaluator
        evaluator = LanguageModelEvaluator(model, vocab, args.device)
        
        # Create test data loader
        print("Creating test data loader...")
        test_loader = create_test_dataloader(args.data_dir, vocab, args.batch_size)
        
        print("\n🔬 LANGUAGE MODEL EVALUATION")
        print("=" * 50)
        
        # Run Comprehensive Evaluation
        results = evaluator.evaluate_model(test_loader, num_generation_samples=100)
        
        # Print all results
        evaluator.print_results(results)
        
        # Generate 3 sample texts for a quick qualitative check
        print(f"\n📝 ADDITIONAL SAMPLE GENERATIONS:")
        generate_sample_texts(model, num_samples=3)
        
        # Save results
        evaluator.save_results(results, os.path.join(args.results_dir, 'evaluation_results.json'))
        
        print(f"\n✅ Evaluation completed successfully!")
        print(f"Results saved to: {os.path.join(args.results_dir, 'evaluation_results.json')}")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {str(e)}")
        raise


def generate_sample_texts(model: LanguageModel, num_samples: int = 5) -> None:
    """
    Generate sample texts using the trained model (from test.py).
    
    Args:
        model: Trained language model
        num_samples: Number of sample texts to generate
    """
    print(f"Generating {num_samples} sample texts:")
    print("=" * 60)
    
    sample_prompts = [
        ['the', 'movie', 'was'],
        ['i', 'really', 'liked'],
        ['this', 'film', 'is'],
        ['the', 'acting', 'was'],
        ['the', 'story', 'is'],
        ['this', 'is', 'one'],
        ['i', 'would', 'recommend'],
        ['the', 'director', 'did'],
        ['this', 'movie', 'has'],
        ['the', 'plot', 'was']
    ]
    
    for i in range(min(num_samples, len(sample_prompts))):
        prompt = sample_prompts[i]
        generated_text = model.generate_text(prompt, max_length=30, temperature=0.8)
        print(f"Sample {i+1}:")
        print(f"Prompt: {' '.join(prompt)}")
        print(f"Generated: {generated_text}")
        print("-" * 50)


if __name__ == "__main__":
    main()
