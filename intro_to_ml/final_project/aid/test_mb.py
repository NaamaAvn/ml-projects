#!/usr/bin/env python3
"""
Test script for MB dataset neural network training
"""

from intro_to_ml.final_project.aid.final_project import load_mb_data, NeuralNetwork
import numpy as np

def test_mb_dataset():
    """Test the MB dataset with a simple neural network"""
    print("=" * 60)
    print("TESTING MB DATASET NEURAL NETWORK")
    print("=" * 60)
    
    # Load MB data
    X_train, X_val, y_train, y_val = load_mb_data()
    
    print(f"\nDataset Summary:")
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(np.unique(y_train))}")
    print(f"Class distribution - Control: {np.sum(y_train == 0)}, Fibrosis: {np.sum(y_train == 1)}")
    
    # Create a simple neural network for testing
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    # Simple architecture for quick testing
    architecture = [input_size, 128, 64, num_classes]
    
    print(f"\nNeural Network Architecture: {architecture}")
    print(f"Total parameters: {input_size * 128 + 128 * 64 + 64 * num_classes + 128 + 64 + num_classes:,}")
    
    # Create and train network
    nn = NeuralNetwork(
        layer_sizes=architecture,
        learning_rate=0.001,
        epochs=50,
        batch_size=16
    )
    
    print(f"\nTraining neural network...")
    training_time = nn.fit(X_train, y_train, verbose=True)
    
    # Evaluate
    train_accuracy = nn.score(X_train, y_train)
    val_accuracy = nn.score(X_val, y_val)
    
    print(f"\nResults:")
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"Training Time: {training_time:.2f} seconds")
    
    # Example predictions
    print(f"\nExample predictions on validation set:")
    sample_indices = np.random.choice(len(X_val), min(5, len(X_val)), replace=False)
    for i, idx in enumerate(sample_indices):
        prediction = nn.predict(X_val[idx:idx+1])
        true_label = y_val[idx]
        pred_class = "Fibrosis" if prediction[0] == 1 else "Control"
        true_class = "Fibrosis" if true_label == 1 else "Control"
        print(f"Sample {i+1}: Predicted {pred_class}, True {true_class}")
    
    print(f"\nTest completed successfully!")
    return nn, val_accuracy

if __name__ == "__main__":
    test_mb_dataset() 