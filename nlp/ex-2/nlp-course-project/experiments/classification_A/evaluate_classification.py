#!/usr/bin/env python3
"""
Classification Evaluation Script

This script evaluates a trained classification model:
- Loads a trained model and training history.
- Plots the training vs. validation loss and accuracy.
- Evaluates the model on the test set.
- Generates and saves a confusion matrix.
- Performs error analysis by saving misclassified examples.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import argparse
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Import from other scripts in the project
from setup_classification_model import (
    LanguageModel, SimpleVocab,
    ClassificationModel,
    load_pretrained_language_model, create_classification_model
)

class IMDBClassificationDataset(torch.utils.data.Dataset):
    """Custom Dataset for IMDB text classification."""
    def __init__(self, data_file: str, vocab: Any, max_length: int):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.vocab = vocab
        self.max_length = max_length
        self.itos = {i: s for s, i in vocab.stoi.items()}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        text, label = self.data[idx]
        
        tokens = text.split()
        indices = [self.vocab[token] for token in tokens]
        
        if len(indices) < self.max_length:
            indices.extend([self.vocab['<pad>']] * (self.max_length - len(indices)))
        else:
            indices = indices[:self.max_length]
        
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return torch.tensor(indices), label_tensor, text

def plot_training_history(history: Dict[str, List[float]], save_path: str):
    """Plot training history and save to file."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy', color='blue')
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='red')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_path}")

def evaluate_and_analyze(model: ClassificationModel, test_loader: DataLoader,
                         device: str, output_dir: str):
    """Evaluate model, plot confusion matrix, and perform error analysis."""
    model.eval()
    all_predictions = []
    all_labels = []
    misclassified_samples = []
    
    with torch.no_grad():
        for input_sequences, labels, original_texts in test_loader:
            input_sequences = input_sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(input_sequences)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            for i in range(len(predicted)):
                if predicted[i] != labels[i]:
                    misclassified_samples.append({
                        'text': original_texts[i],
                        'true_label': labels[i].item(),
                        'predicted_label': predicted[i].item()
                    })

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['neg', 'pos'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Error Analysis
    error_analysis_path = os.path.join(output_dir, 'error_analysis.json')
    with open(error_analysis_path, 'w') as f:
        json.dump(misclassified_samples, f, indent=2)
    print(f"Error analysis report saved to {error_analysis_path}")
    
    # Calculate and print metrics
    from sklearn.metrics import classification_report
    report = classification_report(all_labels, all_predictions, target_names=['neg', 'pos'], output_dict=True)
    print("Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=['neg', 'pos']))

    results_path = os.path.join(output_dir, 'classification_test_results.json')
    with open(results_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Test results saved to {results_path}")


def main(model_dir: str, data_dir: str, output_dir: str,
         trained_model_file: str, max_length: int, device: str):

    print("Evaluating Classification Model")
    print("=" * 80)
    
    # Load vocabulary
    vocab_path = os.path.join(model_dir.replace('model', 'data'), 'vocab.json')
    with open(vocab_path, 'r') as f:
        vocab_stoi = json.load(f)
    vocab = SimpleVocab(vocab_stoi)
    
    # Load test data
    test_file = os.path.join(data_dir, 'test.json')
    test_dataset = IMDBClassificationDataset(test_file, vocab, max_length)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load trained model checkpoint
    checkpoint_path = os.path.join(output_dir, trained_model_file)
    if not os.path.exists(checkpoint_path):
        print(f"Trained model not found at {checkpoint_path}")
        return
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Plot training history
    history = checkpoint.get('history')
    if history:
        history_plot_path = os.path.join(output_dir, 'training_history.png')
        plot_training_history(history, history_plot_path)
    else:
        print("No training history found in checkpoint.")

    # Re-create model architecture
    lm_path = os.path.join(model_dir, 'language_model.pth')
    language_model = load_pretrained_language_model(lm_path, vocab, device)
    
    config_path = os.path.join(output_dir, 'classification_model_config.json')
    with open(config_path, 'r') as f:
        model_config = json.load(f)

    classification_model = create_classification_model(
        language_model=language_model,
        hidden_dims=model_config['hidden_dims'],
        num_classes=model_config['num_classes'],
        pooling_method=model_config['pooling_method'],
        dropout=model_config['dropout']
    ).to(device)
    
    classification_model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate and analyze
    evaluate_and_analyze(classification_model, test_loader, device, output_dir)
    
    print("\nEvaluation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Classification Model")
    parser.add_argument('--model-dir', type=str, default='../../model/',
                       help='Directory containing the pre-trained language model')
    parser.add_argument('--data-dir', type=str, default='../../data/processed_classification_data/',
                       help='Directory containing classification data')
    parser.add_argument('--output-dir', type=str, default='./',
                       help='Directory containing trained model and for saving evaluation outputs')
    parser.add_argument('--trained-model-file', type=str, default='trained_classification_model.pth',
                       help='Filename of the trained classification model')
    parser.add_argument('--max-length', type=int, default=200,
                       help='Maximum sequence length')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run evaluation on')
    
    args = parser.parse_args()
    
    main(
        model_dir=args.model_dir,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        trained_model_file=args.trained_model_file,
        max_length=args.max_length,
        device=args.device
    ) 