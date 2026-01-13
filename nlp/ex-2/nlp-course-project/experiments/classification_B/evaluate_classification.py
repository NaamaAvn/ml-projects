#!/usr/bin/env python3
"""
Classification Evaluation Script - Step 4: Evaluation and Analysis

This script implements step 4 of the text classification pipeline:
- Loads the trained RNN classification model
- Evaluates the model on the test set
- Generates comprehensive evaluation metrics and visualizations
- Performs detailed error analysis on misclassified examples
- Saves all results and analysis

This is the evaluation script for Experiment B: RNN with Word2Vec embeddings.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import os
import argparse
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
import datetime

# Import the RNN classifier from step 2
from setup_classification_model import (
    SimpleVocab, IMDBDataset, RNNClassifier, create_rnn_classifier, evaluate_model
)

class IMDBClassificationDataset(torch.utils.data.Dataset):
    """Custom Dataset for IMDB text classification with original text preservation."""
    def __init__(self, data_file: str, vocab: SimpleVocab, max_length: int):
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
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
        
        return torch.tensor(indices), label_tensor, text

def load_trained_model(model_path: str, config: Dict[str, Any], vocab: SimpleVocab, device: str = 'cpu') -> RNNClassifier:
    """
    Load the trained RNN classifier model.
    
    Args:
        model_path: Path to the trained model file
        config: Model configuration
        vocab: Vocabulary object
        device: Device to load the model on
        
    Returns:
        Loaded RNNClassifier model
    """
    print(f"Loading trained model from: {model_path}")
    
    # Create model with the same configuration
    model = create_rnn_classifier(
        vocab=vocab,
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config['dropout'],
        rnn_type=config['rnn_type'],
        bidirectional=config['bidirectional'],
        device=device
    )
    
    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded successfully!")
    print(f"Best validation accuracy: {checkpoint.get('val_acc', 'N/A')}")
    print(f"Training accuracy: {checkpoint.get('train_acc', 'N/A')}")
    
    return model

def create_test_dataloader(data_dir: str, vocab: SimpleVocab, batch_size: int = 32, 
                          max_length: int = 200) -> DataLoader:
    """
    Create test dataloader for evaluation.
    
    Args:
        data_dir: Directory containing processed classification data
        vocab: Vocabulary object
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        
    Returns:
        Test DataLoader
    """
    test_file = os.path.join(data_dir, 'test.json')
    
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    # Create test dataset
    test_dataset = IMDBClassificationDataset(test_file, vocab, max_length)
    
    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Test dataloader created:")
    print(f"  - Test samples: {len(test_dataset)}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Max sequence length: {max_length}")
    
    return test_loader

def get_predictions_and_probabilities_with_texts(model: RNNClassifier, data_loader: DataLoader, 
                                           device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Get predictions, probabilities, and original texts from the model.
    
    Args:
        model: Trained RNN classifier
        data_loader: Test data loader
        device: Device to run inference on
        
    Returns:
        Tuple of (y_true, y_pred, probabilities, original_texts)
    """
    model.eval()
    y_true = []
    y_pred = []
    probabilities = []
    original_texts = []
    
    with torch.no_grad():
        for input_sequences, labels, texts in tqdm(data_loader, desc="Evaluating"):
            input_sequences = input_sequences.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(input_sequences)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Get probabilities
            probs = torch.softmax(outputs, dim=1)
            
            # Store results
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            probabilities.extend(probs.cpu().numpy())
            original_texts.extend(texts)
    
    return np.array(y_true), np.array(y_pred), np.array(probabilities), original_texts

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {save_path}")


def generate_detailed_report(y_true: np.ndarray, y_pred: np.ndarray, 
                           probabilities: np.ndarray, save_path: str = None) -> Dict[str, float]:
    """
    Generate detailed evaluation report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        probabilities: Predicted probabilities
        save_path: Path to save the report (optional, if None, don't save individual file)
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Calculate basic metrics
    accuracy = (y_true == y_pred).mean()
    
    # Get classification report
    report = classification_report(y_true, y_pred, target_names=['Negative', 'Positive'], 
                                 output_dict=True)
    
    # Calculate ROC AUC
    y_scores = probabilities[:, 1]
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # Calculate average precision
    avg_precision = average_precision_score(y_true, y_scores)
    
    # Compile metrics
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'precision_negative': report['Negative']['precision'],
        'recall_negative': report['Negative']['recall'],
        'f1_negative': report['Negative']['f1-score'],
        'precision_positive': report['Positive']['precision'],
        'recall_positive': report['Positive']['recall'],
        'f1_positive': report['Positive']['f1-score'],
        'macro_avg_precision': report['macro avg']['precision'],
        'macro_avg_recall': report['macro avg']['recall'],
        'macro_avg_f1': report['macro avg']['f1-score'],
        'weighted_avg_precision': report['weighted avg']['precision'],
        'weighted_avg_recall': report['weighted avg']['recall'],
        'weighted_avg_f1': report['weighted avg']['f1-score']
    }
    
    # Save detailed report only if save_path is provided
    if save_path:
        report_data = {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        with open(save_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"Detailed evaluation report saved to: {save_path}")
    
    return metrics

def print_summary_metrics(metrics: Dict[str, float]):
    """
    Print summary of evaluation metrics.
    
    Args:
        metrics: Dictionary containing evaluation metrics
    """
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"Average Precision: {metrics['average_precision']:.4f}")
    print()
    print("Per-Class Metrics:")
    print(f"  Negative Class:")
    print(f"    Precision: {metrics['precision_negative']:.4f}")
    print(f"    Recall: {metrics['recall_negative']:.4f}")
    print(f"    F1-Score: {metrics['f1_negative']:.4f}")
    print(f"  Positive Class:")
    print(f"    Precision: {metrics['precision_positive']:.4f}")
    print(f"    Recall: {metrics['recall_positive']:.4f}")
    print(f"    F1-Score: {metrics['f1_positive']:.4f}")
    print()
    print("Macro Averages:")
    print(f"  Precision: {metrics['macro_avg_precision']:.4f}")
    print(f"  Recall: {metrics['macro_avg_recall']:.4f}")
    print(f"  F1-Score: {metrics['macro_avg_f1']:.4f}")
    print("="*60)

def perform_error_analysis(y_true: np.ndarray, y_pred: np.ndarray, 
                         probabilities: np.ndarray, original_texts: List[str],
                         output_dir: str) -> Dict[str, Any]:
    """
    Perform detailed error analysis on misclassified examples.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        probabilities: Prediction probabilities
        original_texts: Original text samples
        output_dir: Directory to save error analysis
        
    Returns:
        Dictionary containing error analysis results
    """
    print("Performing error analysis...")
    
    # Find misclassified examples
    misclassified_indices = np.where(y_true != y_pred)[0]
    misclassified_samples = []
    
    for idx in misclassified_indices:
        sample = {
            'text': original_texts[idx],
            'true_label': int(y_true[idx]),
            'predicted_label': int(y_pred[idx]),
            'confidence': float(np.max(probabilities[idx])),
            'true_class_probability': float(probabilities[idx][y_true[idx]]),
            'predicted_class_probability': float(probabilities[idx][y_pred[idx]]),
            'text_length': len(original_texts[idx].split()),
            'index': int(idx)
        }
        misclassified_samples.append(sample)
    
    # Analyze error patterns
    error_analysis = {
        'total_samples': len(y_true),
        'misclassified_samples': len(misclassified_samples),
        'error_rate': len(misclassified_samples) / len(y_true),
        'misclassified_examples': misclassified_samples,
        'error_patterns': {
            'false_positives': len([s for s in misclassified_samples if s['true_label'] == 0 and s['predicted_label'] == 1]),
            'false_negatives': len([s for s in misclassified_samples if s['true_label'] == 1 and s['predicted_label'] == 0])
        },
        'confidence_analysis': {
            'avg_confidence_correct': float(np.mean([np.max(probabilities[i]) for i in range(len(y_true)) if y_true[i] == y_pred[i]])),
            'avg_confidence_incorrect': float(np.mean([np.max(probabilities[i]) for i in range(len(y_true)) if y_true[i] != y_pred[i]]))
        },
        'text_length_analysis': {
            'avg_length_correct': float(np.mean([len(original_texts[i].split()) for i in range(len(y_true)) if y_true[i] == y_pred[i]])),
            'avg_length_incorrect': float(np.mean([len(original_texts[i].split()) for i in range(len(y_true)) if y_true[i] != y_pred[i]]))
        }
    }
    
    # Save detailed error analysis
    error_analysis_path = os.path.join(output_dir, 'error_analysis.json')
    with open(error_analysis_path, 'w') as f:
        json.dump(error_analysis, f, indent=2)
    
    print(f"Error analysis saved to: {error_analysis_path}")
    print(f"Total misclassified samples: {len(misclassified_samples)}")
    print(f"Error rate: {error_analysis['error_rate']:.4f}")
    print(f"False positives: {error_analysis['error_patterns']['false_positives']}")
    print(f"False negatives: {error_analysis['error_patterns']['false_negatives']}")
    
    return error_analysis

def main(model_dir: str = '../../model/',
         data_dir: str = '../../data/processed_classification_data/',
         output_dir: str = './',
         trained_model_file: str = 'trained_rnn_classifier.pth',
         config_file: str = './rnn_classifier_config.json',
         batch_size: int = 32,
         max_length: int = 200,
         device: str = 'cpu'):
    """
    Main function to evaluate the trained RNN classification model.
    
    Args:
        model_dir: Directory containing pre-trained models (not used in this experiment)
        data_dir: Directory containing processed classification data
        output_dir: Directory to save evaluation results
        trained_model_file: Name of the trained model file
        config_file: Path to the model configuration file
        batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        device: Device to run evaluation on
    """
    print("Evaluating RNN Classification Model (Experiment B)")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model configuration
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded model configuration: {config}")
    
    # Load vocabulary
    vocab_file = os.path.join(os.path.dirname(config_file), config['vocab_file'])
    with open(vocab_file, 'r') as f:
        vocab_dict = json.load(f)
    
    vocab = SimpleVocab(vocab_dict)
    print(f"Loaded vocabulary with {len(vocab)} words")
    
    # Load trained model
    model_path = os.path.join(output_dir, trained_model_file)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Trained model file not found: {model_path}")
    
    model = load_trained_model(model_path, config, vocab, device)
    
    # Create test dataloader with original texts
    test_loader = create_test_dataloader(data_dir, vocab, batch_size, max_length)
    
    # Get predictions, probabilities, and original texts
    print("\nGenerating predictions and performing error analysis...")
    y_true, y_pred, probabilities, original_texts = get_predictions_and_probabilities_with_texts(model, test_loader, device)
    
    # Perform error analysis
    error_analysis = perform_error_analysis(y_true, y_pred, probabilities, original_texts, output_dir)
    
    # Generate plots
    print("\nGenerating evaluation plots...")
    
    # Confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(y_true, y_pred, cm_path)
    
    # Generate detailed report
    print("\nGenerating detailed evaluation report...")
    metrics = generate_detailed_report(y_true, y_pred, probabilities, None)  # Don't save individual file
    
    # Print summary
    print_summary_metrics(metrics)
    
    # Load training history and test results if they exist
    training_history = {}
    test_results = {}
    
    # Try to load training history
    history_path = os.path.join(output_dir, 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            training_history = json.load(f)
    
    # Try to load test results from training
    test_results_path = os.path.join(output_dir, 'test_results.json')
    if os.path.exists(test_results_path):
        with open(test_results_path, 'r') as f:
            test_results = json.load(f)
    
    # Create unified evaluation results
    unified_results = {
        'experiment_info': {
            'experiment_name': 'Classification_B',
            'model_type': 'RNN Classifier with Word2Vec embeddings',
            'evaluation_timestamp': str(datetime.datetime.now()),
            'model_config': config
        },
        'training_results': {
            'training_history': training_history,
            'test_results_from_training': test_results
        },
        'evaluation_results': {
            'metrics': metrics,
            'classification_report': {
                'Negative': {
                    'precision': metrics['precision_negative'],
                    'recall': metrics['recall_negative'],
                    'f1-score': metrics['f1_negative'],
                    'support': 12500.0
                },
                'Positive': {
                    'precision': metrics['precision_positive'],
                    'recall': metrics['recall_positive'],
                    'f1-score': metrics['f1_positive'],
                    'support': 12500.0
                },
                'accuracy': metrics['accuracy'],
                'macro_avg': {
                    'precision': metrics['macro_avg_precision'],
                    'recall': metrics['macro_avg_recall'],
                    'f1-score': metrics['macro_avg_f1'],
                    'support': 25000.0
                },
                'weighted_avg': {
                    'precision': metrics['weighted_avg_precision'],
                    'recall': metrics['weighted_avg_recall'],
                    'f1-score': metrics['weighted_avg_f1'],
                    'support': 25000.0
                }
            },
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        },
        'error_analysis': error_analysis,
        'data_info': {
            'test_samples': len(y_true),
            'class_distribution': {
                'negative': int(np.sum(y_true == 0)),
                'positive': int(np.sum(y_true == 1))
            }
        }
    }
    
    # Save unified evaluation results
    unified_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(unified_path, 'w') as f:
        json.dump(unified_results, f, indent=2)
    
    print(f"\nUnified evaluation results saved to: {unified_path}")
    print("\nEvaluation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate RNN classification model with Word2Vec embeddings')
    parser.add_argument('--model-dir', type=str, default='../../model/',
                        help='Directory containing pre-trained models (not used in this experiment)')
    parser.add_argument('--data-dir', type=str, default='../../data/processed_classification_data/',
                        help='Directory containing processed classification data')
    parser.add_argument('--output-dir', type=str, default='./',
                        help='Directory to save evaluation results')
    parser.add_argument('--trained-model-file', type=str, default='trained_rnn_classifier.pth',
                        help='Name of the trained model file')
    parser.add_argument('--config-file', type=str, default='./rnn_classifier_config.json',
                        help='Path to the model configuration file')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
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
        config_file=args.config_file,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device
    ) 