#!/usr/bin/env python3
"""
Demo script for the Language Model implementation.

This script demonstrates how to:
1. Load a trained language model
2. Generate text samples
3. Evaluate the model on test data
4. Analyze model performance
"""

import torch
import json
import os
from process_imdb_lm import LanguageModel, load_and_test_model, build_vocab_and_dataloaders
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import IMDB
import matplotlib.pyplot as plt
import numpy as np

def demo_text_generation(model: LanguageModel, num_samples: int = 5):
    """
    Demonstrate text generation with the language model.
    
    Args:
        model: Trained language model
        num_samples: Number of text samples to generate
    """
    print("\n" + "="*60)
    print("TEXT GENERATION DEMO")
    print("="*60)
    
    # Different types of prompts for variety
    prompts = [
        ['the', 'movie', 'was'],
        ['i', 'really', 'liked'],
        ['this', 'film', 'is'],
        ['the', 'acting', 'was'],
        ['the', 'story', 'is'],
        ['i', 'hated', 'this'],
        ['the', 'director', 'did'],
        ['this', 'is', 'one'],
        ['the', 'best', 'movie'],
        ['the', 'worst', 'film']
    ]
    
    temperatures = [0.5, 0.8, 1.0, 1.2]
    
    for i in range(min(num_samples, len(prompts))):
        prompt = prompts[i]
        print(f"\nSample {i+1}:")
        print(f"Prompt: {' '.join(prompt)}")
        
        for temp in temperatures:
            generated_text = model.generate_text(
                start_tokens=prompt,
                max_length=25,
                temperature=temp
            )
            print(f"Temperature {temp}: {generated_text}")
        print("-" * 50)

def analyze_model_performance(model: LanguageModel, test_loader):
    """
    Analyze the model's performance in detail.
    
    Args:
        model: Trained language model
        test_loader: DataLoader for test data
    """
    print("\n" + "="*60)
    print("MODEL PERFORMANCE ANALYSIS")
    print("="*60)
    
    model.eval()
    device = next(model.parameters()).device
    
    # Calculate perplexity
    total_loss = 0.0
    total_tokens = 0
    word_predictions = []
    
    with torch.no_grad():
        for batch_idx, (input_sequences, target_sequences) in enumerate(test_loader):
            if batch_idx >= 10:  # Limit for faster analysis
                break
                
            input_sequences = input_sequences.to(device)
            target_sequences = target_sequences.to(device)
            
            output_logits, _ = model.forward(input_sequences)
            loss = model.calculate_loss(output_logits, target_sequences)
            
            # Calculate perplexity
            batch_size, seq_length, vocab_size = output_logits.shape
            output_logits = output_logits.view(-1, vocab_size)
            targets = target_sequences.view(-1)
            
            # Count non-padding tokens
            mask = targets != model.vocab['<pad>']
            total_tokens += mask.sum().item()
            total_loss += loss.item() * mask.sum().item()
            
            # Store some predictions for analysis
            if batch_idx < 3:
                predicted = torch.argmax(output_logits, dim=1)
                for i in range(min(5, len(targets))):
                    if mask[i]:
                        target_word = model.vocab.get_itos()[targets[i].item()]
                        pred_word = model.vocab.get_itos()[predicted[i].item()]
                        word_predictions.append((target_word, pred_word))
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    
    # Show some prediction examples
    print(f"\nSample Predictions (Target -> Predicted):")
    for i, (target, predicted) in enumerate(word_predictions[:20]):
        correct = "✓" if target == predicted else "✗"
        print(f"{i+1:2d}. {target:15s} -> {predicted:15s} {correct}")
    
    # Calculate accuracy
    correct_predictions = sum(1 for target, pred in word_predictions if target == pred)
    accuracy = correct_predictions / len(word_predictions) if word_predictions else 0
    print(f"\nPrediction Accuracy: {accuracy:.2%}")

def interactive_text_generation(model: LanguageModel):
    """
    Interactive text generation demo.
    
    Args:
        model: Trained language model
    """
    print("\n" + "="*60)
    print("INTERACTIVE TEXT GENERATION")
    print("="*60)
    print("Enter starting words (space-separated) or 'quit' to exit")
    print("Example: 'the movie was'")
    
    while True:
        try:
            user_input = input("\nEnter prompt: ").strip().lower()
            
            if user_input == 'quit':
                break
            
            if not user_input:
                print("Please enter some words.")
                continue
            
            # Parse input
            start_tokens = user_input.split()
            
            # Validate tokens
            valid_tokens = []
            for token in start_tokens:
                if token in model.vocab:
                    valid_tokens.append(token)
                else:
                    print(f"Warning: '{token}' not in vocabulary, using <unk>")
                    valid_tokens.append('<unk>')
            
            if not valid_tokens:
                print("No valid tokens found. Please try again.")
                continue
            
            # Generate text
            print("\nGenerating text...")
            generated_text = model.generate_text(
                start_tokens=valid_tokens,
                max_length=50,
                temperature=0.8
            )
            
            print(f"Generated: {generated_text}")
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """Main demo function."""
    print("Language Model Demo")
    print("="*60)
    
    # Check if model exists
    model_path = './models/language_model.pth'
    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}")
        print("Please run the main training script first:")
        print("python process_imdb_lm.py")
        return
    
    # Load the trained model
    print("Loading trained model...")
    model = load_and_test_model(model_path)
    
    if model is None:
        print("Failed to load model.")
        return
    
    # Load test data for evaluation
    print("\nLoading test data...")
    tokenizer = get_tokenizer('basic_english')
    test_iter = IMDB(split='test')
    test_data = list(test_iter)
    
    # Build vocabulary and create test loader
    from process_imdb_lm import build_vocab_and_dataloaders
    vocab, _, _, test_loader = build_vocab_and_dataloaders(
        test_data[:1000],  # Use subset for faster demo
        tokenizer,
        batch_size=16
    )
    
    # Run demos
    demo_text_generation(model, num_samples=5)
    analyze_model_performance(model, test_loader)
    
    # Interactive demo
    try:
        interactive_text_generation(model)
    except Exception as e:
        print(f"Interactive demo failed: {e}")
    
    print("\nDemo completed!")

if __name__ == '__main__':
    main() 