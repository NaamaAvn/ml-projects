#!/usr/bin/env python3
"""
Simple test script to verify the evaluation functionality.
"""

import os
import sys
import json
from evaluate import LanguageModelEvaluator, load_model_and_vocab, create_test_dataloader

def test_evaluation():
    """Test the evaluation script with a small subset of data."""
    
    print("🧪 Testing Language Model Evaluation...")
    
    # Check if required files exist
    model_path = './model/language_model.pth'
    vocab_path = './data/vocab.json'
    data_dir = './data/'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(vocab_path):
        print(f"❌ Vocabulary file not found: {vocab_path}")
        return False
    
    if not os.path.exists(os.path.join(data_dir, 'processed_lm_data', 'test.json')):
        print(f"❌ Test data file not found: {os.path.join(data_dir, 'processed_lm_data', 'test.json')}")
        return False
    
    try:
        # Load model and vocabulary
        print("📥 Loading model and vocabulary...")
        model, vocab = load_model_and_vocab(model_path, vocab_path, device='cpu')
        print(f"✅ Model loaded successfully (vocab size: {len(vocab)})")
        
        # Create test data loader with small batch size for testing
        print("📊 Creating test data loader...")
        test_loader = create_test_dataloader(data_dir, vocab, batch_size=8)
        print(f"✅ Test data loader created")
        
        # Create evaluator
        print("🔧 Creating evaluator...")
        evaluator = LanguageModelEvaluator(model, vocab, device='cpu')
        print("✅ Evaluator created successfully")
        
        # Test perplexity calculation with a small subset
        print("📈 Testing perplexity calculation...")
        perplexity = evaluator.calculate_perplexity(test_loader)
        print(f"✅ Perplexity calculated: {perplexity:.4f}")
        
        # Test accuracy calculation
        print("🎯 Testing accuracy calculation...")
        accuracy = evaluator.calculate_accuracy(test_loader)
        print(f"✅ Accuracy calculated: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Test text generation (small number of samples)
        print("✍️ Testing text generation...")
        generated_samples = evaluator.generate_text_samples(num_samples=5, max_length=20)
        print(f"✅ Generated {len(generated_samples)} text samples")
        
        # Test reference sample extraction
        print("📝 Testing reference sample extraction...")
        reference_samples = evaluator.get_reference_samples(test_loader, num_samples=5)
        print(f"✅ Extracted {len(reference_samples)} reference samples")
        
        # Test BLEU calculation
        print("🎯 Testing BLEU score calculation...")
        bleu_scores = evaluator.calculate_bleu_scores(generated_samples, reference_samples)
        print(f"✅ BLEU scores calculated: {bleu_scores}")
        
        # Test ROUGE calculation
        print("🔍 Testing ROUGE score calculation...")
        rouge_scores = evaluator.calculate_rouge_scores(generated_samples, reference_samples)
        print(f"✅ ROUGE scores calculated")
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_evaluation()
    sys.exit(0 if success else 1) 