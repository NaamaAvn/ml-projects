#!/usr/bin/env python3
"""
MB Dataset Neural Network Experiment
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import json
import random
import time
from datetime import datetime

# Add the parent directory to the path to import the neural network
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from neural_network import NeuralNetwork

def load_mb_data():
    """Load and preprocess MB dataset from inputs directory"""
    print("Loading MB dataset from inputs directory...")
    
    try:
        import pandas as pd
        
        # Load training data
        print("Loading MB training data...")
        train_data = pd.read_csv("inputs/MB_data_train.csv", index_col=0)
        print(f"MB training data shape: {train_data.shape}")
        
        # Extract labels from row names
        # Pt_Fibro_* -> class 1 (Fibrosis)
        # Pt_Ctrl_* -> class 0 (Control)
        y_train = []
        for patient_id in train_data.index:
            if patient_id.startswith('Pt_Fibro_'):
                y_train.append(1)  # Fibrosis
            elif patient_id.startswith('Pt_Ctrl_'):
                y_train.append(0)  # Control
            else:
                print(f"Warning: Unknown patient type: {patient_id}")
                y_train.append(0)  # Default to control
        
        y_train = np.array(y_train)
        X_train = train_data.values
        
        # Normalize features (standard scaling)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        # Split into train and validation sets (since we only have one file)
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"Training set: {X_train_split.shape[0]} samples, {X_train_split.shape[1]} features")
        print(f"Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features")
        print(f"Number of classes: {len(np.unique(y_train))}")
        print(f"Class distribution - Control: {np.sum(y_train == 0)}, Fibrosis: {np.sum(y_train == 1)}")
        
        return X_train_split, X_val, y_train_split, y_val
        
    except FileNotFoundError as e:
        print(f"Error: Could not find MB data file in inputs directory: {e}")
        print("Falling back to sklearn breast cancer dataset...")
        
        # Fallback to breast cancer dataset (similar binary classification)
        cancer = load_breast_cancer()
        X, y = cancer.data, cancer.target
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Using sklearn breast cancer dataset (similar binary classification)")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        return X_train, X_val, y_train, y_val
        
    except Exception as e:
        print(f"Error loading MB data: {e}")
        print("Falling back to sklearn breast cancer dataset...")
        
        # Fallback to breast cancer dataset
        cancer = load_breast_cancer()
        X, y = cancer.data, cancer.target
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Using sklearn breast cancer dataset (similar binary classification)")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Number of classes: {len(np.unique(y))}")
        
        return X_train, X_val, y_train, y_val

def optimize_hyperparameters(X_train, y_train, X_val, y_val, max_trials=25):
    """
    Optimize hyperparameters for the neural network using smart sampling
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        max_trials: Maximum number of hyperparameter combinations to try
    
    Returns:
        best_config: Best hyperparameter configuration
        results: List of all trial results
    """
    print("=" * 60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Define hyperparameter search space
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    # Different network architectures to try (various depths and widths)
    architectures = [
        [input_size, 64, 32, num_classes],           # Small network
        [input_size, 128, 64, num_classes],          # Medium network
        [input_size, 256, 128, 64, num_classes],     # Large network
        [input_size, 128, 128, num_classes],         # Wide network
        [input_size, 64, 64, 64, num_classes],       # Deep network
        [input_size, 512, 256, 128, 64, num_classes], # Very large network
        [input_size, 32, 32, 32, 32, num_classes],   # Very deep network
        [input_size, 256, 256, num_classes],         # Very wide network
    ]
    
    # Learning rates to examine (smaller for high-dimensional data)
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    
    # Epochs to examine
    epochs_list = [20, 30, 50, 100, 150]
    
    # Batch sizes to examine (smaller for smaller dataset)
    batch_sizes = [4, 8, 16, 32, 64]
    
    # Smart sampling strategy
    def generate_smart_combinations():
        """Generate combinations ensuring good coverage of all parameters"""
        combinations = []
        
        # Strategy 1: Test each architecture with good default parameters
        print("Phase 1: Testing each architecture with default parameters...")
        for i, arch in enumerate(architectures):
            combinations.append({
                'architecture': arch,
                'learning_rate': 0.001,  # Good default for high-dimensional data
                'epochs': 50,           # Good default
                'batch_size': 16,       # Good default for smaller dataset
                'phase': 1,
                'description': f"Architecture {i+1} with defaults"
            })
        
        # Strategy 2: Test learning rates with best architecture so far
        print("Phase 2: Testing learning rates with best architecture...")
        best_arch_idx = 1  # Start with medium architecture
        for lr in learning_rates:
            combinations.append({
                'architecture': architectures[best_arch_idx],
                'learning_rate': lr,
                'epochs': 50,
                'batch_size': 16,
                'phase': 2,
                'description': f"Learning rate {lr} with best arch"
            })
        
        # Strategy 3: Test epochs with best architecture and learning rate
        print("Phase 3: Testing epochs with best arch and LR...")
        best_lr = 0.001
        for epochs in epochs_list:
            combinations.append({
                'architecture': architectures[best_arch_idx],
                'learning_rate': best_lr,
                'epochs': epochs,
                'batch_size': 16,
                'phase': 3,
                'description': f"Epochs {epochs} with best arch and LR"
            })
        
        # Strategy 4: Test batch sizes with best configuration so far
        print("Phase 4: Testing batch sizes with best config...")
        best_epochs = 50
        for batch_size in batch_sizes:
            combinations.append({
                'architecture': architectures[best_arch_idx],
                'learning_rate': best_lr,
                'epochs': best_epochs,
                'batch_size': batch_size,
                'phase': 4,
                'description': f"Batch size {batch_size} with best config"
            })
        
        # Strategy 5: Random combinations to fill remaining trials
        remaining_trials = max_trials - len(combinations)
        if remaining_trials > 0:
            print(f"Phase 5: Testing {remaining_trials} random combinations...")
            for i in range(remaining_trials):
                arch = random.choice(architectures)
                lr = random.choice(learning_rates)
                epochs = random.choice(epochs_list)
                batch_size = random.choice(batch_sizes)
                
                combinations.append({
                    'architecture': arch,
                    'learning_rate': lr,
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'phase': 5,
                    'description': f"Random combination {i+1}"
                })
        
        return combinations
    
    # Generate smart combinations
    combinations = generate_smart_combinations()
    
    results = []
    best_accuracy = 0
    best_config = None
    
    # Create plots directory if it doesn't exist
    plots_dir = os.path.join(os.path.dirname(__file__), "outputs", "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    print(f"Testing {len(architectures)} architectures")
    print(f"Testing {len(learning_rates)} learning rates: {learning_rates}")
    print(f"Testing {len(epochs_list)} epoch values: {epochs_list}")
    print(f"Testing {len(batch_sizes)} batch sizes: {batch_sizes}")
    print(f"Smart sampling strategy: {len(combinations)} combinations")
    
    for trial_count, combo in enumerate(combinations, 1):
        print(f"\n--- Trial {trial_count}/{len(combinations)} (Phase {combo['phase']}) ---")
        print(f"Description: {combo['description']}")
        print(f"Architecture: {combo['architecture']}")
        print(f"Learning Rate: {combo['learning_rate']}")
        print(f"Epochs: {combo['epochs']}")
        print(f"Batch Size: {combo['batch_size']}")
        
        # Create and train network
        try:
            nn = NeuralNetwork(
                layer_sizes=combo['architecture'],
                learning_rate=combo['learning_rate'],
                epochs=combo['epochs'],
                batch_size=combo['batch_size']
            )
            
            # Train with reduced verbosity
            training_time = nn.fit(X_train, y_train, verbose=False)
            
            # Evaluate
            train_accuracy = nn.score(X_train, y_train)
            val_accuracy = nn.score(X_val, y_val)
            
            # Store results
            result = {
                'trial': trial_count,
                'phase': combo['phase'],
                'description': combo['description'],
                'architecture': combo['architecture'],
                'learning_rate': combo['learning_rate'],
                'epochs': combo['epochs'],
                'batch_size': combo['batch_size'],
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'training_time': training_time,
                'final_loss': nn.training_loss[-1] if nn.training_loss else None,
                'total_params': sum(w.size + b.size for w, b in zip(nn.weights, nn.biases)),
                'architecture_depth': len(combo['architecture']) - 2,  # Number of hidden layers
                'architecture_width': max(combo['architecture'][1:-1]) if len(combo['architecture']) > 2 else 0  # Max hidden layer size
            }
            
            results.append(result)
            
            print(f"Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
            print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
            print(f"Training Time: {training_time:.2f}s")
            print(f"Total Parameters: {result['total_params']:,}")
            print(f"Architecture Depth: {result['architecture_depth']} layers")
            print(f"Architecture Width: {result['architecture_width']} neurons")
            
            # Update best configuration
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_config = result.copy()
                print("*** NEW BEST CONFIGURATION! ***")
                
                # Update best parameters for next phases
                if combo['phase'] == 1:
                    # Update best architecture index
                    best_arch_idx = architectures.index(combo['architecture'])
                elif combo['phase'] == 2:
                    # Update best learning rate
                    best_lr = combo['learning_rate']
                elif combo['phase'] == 3:
                    # Update best epochs
                    best_epochs = combo['epochs']
            
        except Exception as e:
            print(f"Error in trial {trial_count}: {str(e)}")
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Total trials completed: {len(results)}")
    print(f"Best validation accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print("\nBest configuration:")
    for key, value in best_config.items():
        if key not in ['trial', 'phase', 'description']:
            print(f"  {key}: {value}")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create optimization results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), "outputs", "optimization_results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    results_file = os.path.join(results_dir, f"optimization_results_{timestamp}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'best_config': best_config,
            'all_results': results,
            'sampling_strategy': 'smart_phased',
            'summary': {
                'total_trials': len(results),
                'best_accuracy': best_accuracy,
                'dataset_info': {
                    'input_size': input_size,
                    'num_classes': num_classes,
                    'train_samples': len(X_train),
                    'val_samples': len(X_val)
                }
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Generate and save plots
    plot_optimization_results(results, plots_dir, timestamp)
    
    return best_config, results

def plot_optimization_results(results, plots_dir, timestamp):
    """Plot optimization results for analysis and save to plots directory"""
    if not results:
        print("No results to plot")
        return
    
    # Create plots directory if it doesn't exist
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    # Extract data
    trials = [r['trial'] for r in results]
    val_accuracies = [r['val_accuracy'] for r in results]
    train_accuracies = [r['train_accuracy'] for r in results]
    training_times = [r['training_time'] for r in results]
    total_params = [r['total_params'] for r in results]
    learning_rates = [r['learning_rate'] for r in results]
    epochs_list = [r['epochs'] for r in results]
    batch_sizes = [r['batch_size'] for r in results]
    architecture_depths = [r['architecture_depth'] for r in results]
    architecture_widths = [r['architecture_width'] for r in results]
    phases = [r['phase'] for r in results]
    
    # Plot 1: Accuracy vs Trial with phase coloring
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 3, 1)
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for phase in range(1, 6):
        phase_trials = [i+1 for i, p in enumerate(phases) if p == phase]
        phase_accuracies = [acc for i, acc in enumerate(val_accuracies) if phases[i] == phase]
        if phase_trials:
            plt.plot(phase_trials, phase_accuracies, 'o-', color=colors[phase-1], 
                    label=f'Phase {phase}', alpha=0.7)
    plt.xlabel('Trial Number')
    plt.ylabel('Validation Accuracy')
    plt.title('Accuracy vs Trial (by Phase)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Training Time vs Trial
    plt.subplot(2, 3, 2)
    plt.plot(trials, training_times, 'go-', alpha=0.7)
    plt.xlabel('Trial Number')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time vs Trial')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Parameters vs Accuracy
    plt.subplot(2, 3, 3)
    scatter = plt.scatter(total_params, val_accuracies, alpha=0.7, c=phases, cmap='viridis')
    plt.xlabel('Total Parameters')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Complexity vs Performance')
    plt.colorbar(scatter, label='Phase')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Learning Rate vs Accuracy
    plt.subplot(2, 3, 4)
    lr_accuracy = {}
    for lr, acc in zip(learning_rates, val_accuracies):
        if lr not in lr_accuracy:
            lr_accuracy[lr] = []
        lr_accuracy[lr].append(acc)
    
    lr_means = [np.mean(lr_accuracy[lr]) for lr in sorted(lr_accuracy.keys())]
    lr_stds = [np.std(lr_accuracy[lr]) for lr in sorted(lr_accuracy.keys())]
    plt.errorbar(sorted(lr_accuracy.keys()), lr_means, yerr=lr_stds, fmt='o-', capsize=5)
    plt.xlabel('Learning Rate')
    plt.ylabel('Mean Validation Accuracy')
    plt.title('Learning Rate vs Performance')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Epochs vs Accuracy
    plt.subplot(2, 3, 5)
    epochs_accuracy = {}
    for epochs, acc in zip(epochs_list, val_accuracies):
        if epochs not in epochs_accuracy:
            epochs_accuracy[epochs] = []
        epochs_accuracy[epochs].append(acc)
    
    epochs_means = [np.mean(epochs_accuracy[e]) for e in sorted(epochs_accuracy.keys())]
    epochs_stds = [np.std(epochs_accuracy[e]) for e in sorted(epochs_accuracy.keys())]
    plt.errorbar(sorted(epochs_accuracy.keys()), epochs_means, yerr=epochs_stds, fmt='s-', capsize=5)
    plt.xlabel('Epochs')
    plt.ylabel('Mean Validation Accuracy')
    plt.title('Epochs vs Performance')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Batch Size vs Accuracy
    plt.subplot(2, 3, 6)
    batch_accuracy = {}
    for batch, acc in zip(batch_sizes, val_accuracies):
        if batch not in batch_accuracy:
            batch_accuracy[batch] = []
        batch_accuracy[batch].append(acc)
    
    batch_means = [np.mean(batch_accuracy[b]) for b in sorted(batch_accuracy.keys())]
    batch_stds = [np.std(batch_accuracy[b]) for b in sorted(batch_accuracy.keys())]
    plt.errorbar(sorted(batch_accuracy.keys()), batch_means, yerr=batch_stds, fmt='^-', capsize=5)
    plt.xlabel('Batch Size')
    plt.ylabel('Mean Validation Accuracy')
    plt.title('Batch Size vs Performance')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, f"optimization_summary_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Optimization summary plot saved to: {plot_path}")

def load_best_config():
    """Load the best configuration from saved results in outputs/optimization_results directory"""
    import glob
    
    # Look for optimization results in outputs/optimization_results directory
    results_dir = os.path.join(os.path.dirname(__file__), "outputs", "optimization_results")
    if not os.path.exists(results_dir):
        print(f"Optimization results directory not found: {results_dir}")
        return None
    
    # Find all JSON result files in the directory
    result_files = glob.glob(os.path.join(results_dir, "optimization_results_*.json"))
    if not result_files:
        print(f"No optimization result files found in {results_dir}")
        return None
    
    # Get the most recent file
    latest_file = max(result_files, key=os.path.getctime)
    
    try:
        with open(latest_file, 'r') as f:
            data = json.load(f)
            print(f"Loaded best configuration from: {latest_file}")
            print(f"Best accuracy: {data['best_config']['val_accuracy']:.4f}")
            print(f"Optimization timestamp: {data['timestamp']}")
            return data['best_config']
    except Exception as e:
        print(f"Error loading configuration from {latest_file}: {e}")
        return None

def main():
    """Main function to demonstrate the neural network on MB dataset"""
    # Load MB data
    X_train, X_val, y_train, y_val = load_mb_data()
    
    # Try to load best configuration from previous optimization
    best_config = load_best_config()
    
    if best_config is None:
        print("No previous optimization results found. Using default configuration for MB dataset.")
        # Use default configuration optimized for binary classification
        input_size = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        best_config = {
            'architecture': [input_size, 256, 128, 64, num_classes],
            'learning_rate': 0.001,  # Smaller learning rate for high-dimensional data
            'epochs': 100,
            'batch_size': 16  # Smaller batch size for smaller dataset
        }
    
    # Train final model with best configuration
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL ON MB DATASET")
    print("=" * 60)
    
    final_nn = NeuralNetwork(
        layer_sizes=best_config['architecture'],
        learning_rate=best_config['learning_rate'],
        epochs=best_config['epochs'],
        batch_size=best_config['batch_size']
    )
    
    # Train with full verbosity
    final_nn.fit(X_train, y_train, verbose=True)
    
    # Final evaluation on validation set
    final_val_accuracy = final_nn.score(X_val, y_val)
    print(f"\nFinal Validation Accuracy: {final_val_accuracy:.4f} ({final_val_accuracy*100:.2f}%)")
    
    # Additional metrics for binary classification
    from sklearn.metrics import classification_report, confusion_matrix
    predictions = final_nn.predict(X_val)
    print("\nClassification Report:")
    print(classification_report(y_val, predictions, target_names=['Control', 'Fibrosis']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, predictions)
    print(cm)
    
    # Plot training loss for final model and save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plots_dir = os.path.join(os.path.dirname(__file__), "outputs", "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    
    loss_plot_path = os.path.join(plots_dir, f"mb_final_model_training_loss_{timestamp}.png")
    final_nn.plot_training_loss(save_path=loss_plot_path)
    
    # Example predictions
    print("\nExample predictions:")
    sample_indices = np.random.choice(len(X_val), min(5, len(X_val)), replace=False)
    for i, idx in enumerate(sample_indices):
        prediction = final_nn.predict(X_val[idx:idx+1])
        true_label = y_val[idx]
        pred_class = "Fibrosis" if prediction[0] == 1 else "Control"
        true_class = "Fibrosis" if true_label == 1 else "Control"
        print(f"Sample {i+1}: Predicted {pred_class}, True {true_class}")

def run_optimization():
    """Separate function to run hyperparameter optimization"""
    print("Starting hyperparameter optimization on MB dataset...")
    
    # Load MB data
    X_train, X_val, y_train, y_val = load_mb_data()
    
    # Run hyperparameter optimization
    best_config, results = optimize_hyperparameters(X_train, y_train, X_val, y_val, max_trials=25)
    
    print(f"\nOptimization completed! Best configuration found:")
    print(f"Architecture: {best_config['architecture']}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Epochs: {best_config['epochs']}")
    print(f"Batch Size: {best_config['batch_size']}")
    print(f"Best Validation Accuracy: {best_config['val_accuracy']:.4f}")

if __name__ == "__main__":
    # Run optimization if specified, otherwise run main
    if len(sys.argv) > 1:
        if sys.argv[1] == "optimize":
            run_optimization()
        else:
            print("Usage:")
            print("  python mb_experiment.py          # Run MB training")
            print("  python mb_experiment.py optimize # Run MB optimization")
    else:
        main() 