import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate=0.01, epochs=100, batch_size=32):
        """
        Initialize a fully connected feed-forward neural network
        
        Args:
            layer_sizes: List of integers representing the number of neurons in each layer
                         [input_size, hidden_size1, ..., hidden_sizeN, output_size]
            learning_rate: Learning rate for gradient descent
            epochs: Number of training epochs
            batch_size: Size of mini-batches for training
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.weights = []
        self.biases = []
        self.initialize_parameters()
        
    def initialize_parameters(self):
        """Initialize weights and biases for all layers"""
        for i in range(len(self.layer_sizes) - 1):
            # He initialization for better gradient flow
            w = np.random.randn(self.layer_sizes[i+1], self.layer_sizes[i]) * np.sqrt(2.0 / self.layer_sizes[i])
            b = np.zeros((self.layer_sizes[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def sigmoid_derivative(self, z):
        """Derivative of sigmoid function"""
        s = self.sigmoid(z)
        return s * (1 - s)
    
    def relu(self, z):
        """ReLU activation function"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Derivative of ReLU function"""
        return np.where(z > 0, 1, 0)
    
    def softmax(self, z):
        """Softmax activation function for output layer"""
        exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)
    
    def forward_propagation(self, X):
        """Forward propagation through the network"""
        activations = [X]
        z_values = []
        
        # Hidden layers with ReLU activation
        for i in range(len(self.weights) - 1):
            z = np.dot(self.weights[i], activations[-1]) + self.biases[i]
            z_values.append(z)
            activation = self.relu(z)
            activations.append(activation)
        
        # Output layer with softmax activation
        z = np.dot(self.weights[-1], activations[-1]) + self.biases[-1]
        z_values.append(z)
        activation = self.softmax(z)
        activations.append(activation)
        
        return activations, z_values
    
    def backward_propagation(self, X, y, activations, z_values):
        """Backward propagation to compute gradients"""
        m = X.shape[1]
        delta = activations[-1] - y  # Error at output layer
        
        weight_gradients = []
        bias_gradients = []
        
        # Backpropagate through layers
        for i in range(len(self.weights) - 1, -1, -1):
            # Compute gradients for weights and biases
            dW = np.dot(delta, activations[i].T) / m
            db = np.sum(delta, axis=1, keepdims=True) / m
            
            weight_gradients.insert(0, dW)
            bias_gradients.insert(0, db)
            
            # Compute delta for next layer (if not at input layer)
            if i > 0:
                delta = np.dot(self.weights[i].T, delta) * self.relu_derivative(z_values[i-1])
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients, bias_gradients):
        """Update weights and biases using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def one_hot_encode(self, y, num_classes):
        """Convert labels to one-hot encoding"""
        m = y.shape[0]
        y_one_hot = np.zeros((num_classes, m))
        y_one_hot[y, np.arange(m)] = 1
        return y_one_hot
    
    def fit(self, X, y, verbose=True):
        """
        Train the neural network
        
        Args:
            X: Training features (n_features, n_samples)
            y: Training labels (n_samples,)
            verbose: Whether to print training progress
        """
        # Ensure X is in correct format (features x samples)
        if X.shape[0] != self.layer_sizes[0]:
            X = X.T
        
        # One-hot encode the labels
        num_classes = self.layer_sizes[-1]
        y_one_hot = self.one_hot_encode(y, num_classes)
        
        # Training history
        self.training_loss = []
        
        if verbose:
            print(f"Training neural network with {self.epochs} epochs...")
        start_time = time.time()
        
        for epoch in range(self.epochs):
            # Mini-batch training
            indices = np.random.permutation(X.shape[1])
            total_loss = 0
            
            for i in range(0, X.shape[1], self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                X_batch = X[:, batch_indices]
                y_batch = y_one_hot[:, batch_indices]
                
                # Forward propagation
                activations, z_values = self.forward_propagation(X_batch)
                
                # Compute loss (cross-entropy)
                loss = -np.mean(np.sum(y_batch * np.log(activations[-1] + 1e-15), axis=0))
                total_loss += loss
                
                # Backward propagation
                weight_gradients, bias_gradients = self.backward_propagation(X_batch, y_batch, activations, z_values)
                
                # Update parameters
                self.update_parameters(weight_gradients, bias_gradients)
            
            avg_loss = total_loss / (X.shape[1] // self.batch_size + 1)
            self.training_loss.append(avg_loss)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        training_time = time.time() - start_time
        if verbose:
            print(f"Training completed in {training_time:.2f} seconds")
            print(f"Final loss: {self.training_loss[-1]:.4f}")
        
        return training_time
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Args:
            X: Input features (n_features, n_samples) or (n_samples, n_features)
        
        Returns:
            predictions: Predicted class labels
        """
        # Ensure X is in correct format
        if X.shape[0] != self.layer_sizes[0]:
            X = X.T
        
        # Forward propagation
        activations, _ = self.forward_propagation(X)
        
        # Return predicted class (argmax of softmax output)
        predictions = np.argmax(activations[-1], axis=0)
        return predictions
    
    def score(self, X, y):
        """
        Calculate accuracy score
        
        Args:
            X: Test features
            y: True labels
        
        Returns:
            accuracy: Accuracy score between 0 and 1
        """
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        return accuracy
    
    def plot_training_loss(self, save_path=None):
        """Plot the training loss over epochs"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.training_loss)
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training loss plot saved to: {save_path}")
        else:
            plt.show()
        plt.close()

def load_mnist_data():
    """Load and preprocess MNIST dataset"""
        
    # Load training data
    print("Loading training data...")
    train_data = pd.read_csv("./MNIST-train.csv")
    print(f"Training data shape: {train_data.shape}")
    
    # Load test data
    print("Loading test data...")
    test_data = pd.read_csv("./MNIST-test.csv")
    print(f"Test data shape: {test_data.shape}")
    
    # Separate features and labels using 'y' column
    X_train_full = train_data.drop('y', axis=1).values  # All columns except 'y'
    y_train_full = train_data['y'].values               # 'y' column (labels)
    
    X_test = test_data.drop('y', axis=1).values    # All columns except 'y'
    y_test = test_data['y'].values                 # 'y' column (labels)
    
    # Split training data into train and validation sets (80/20 split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    # Normalize pixel values to [0, 1]
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_synthetic_test_set(X_train, y_train, test_size=0.2, random_state=42):
    """
    Create a synthetic test set from training data by splitting it further
    
    Args:
        X_train: Training features
        y_train: Training labels
        test_size: Proportion of data to use for test set
        random_state: Random seed for reproducibility
    
    Returns:
        X_train_new, X_test, y_train_new, y_test: Split data
    """
    from sklearn.model_selection import train_test_split
    
    # Split the training data into new training and test sets
    X_train_new, X_test, y_train_new, y_test = train_test_split(
        X_train, y_train, test_size=test_size, random_state=random_state, stratify=y_train
    )
    
    print(f"Created synthetic test set:")
    print(f"  New training set: {X_train_new.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    print(f"  Test set class distribution: {np.bincount(y_test)}")
    
    return X_train_new, X_test, y_train_new, y_test

def simple_mnist_test():
    """Simple test function to verify NeuralNetwork works on MNIST"""
    print("=" * 60)
    print("SIMPLE MNIST NEURAL NETWORK TEST")
    print("=" * 60)
    
    # Load MNIST data
    print("Loading MNIST dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist_data()
    
    print(f"\nDataset Summary:")
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # Use a smaller subset for faster testing
    print(f"\nUsing smaller subset for faster testing...")
    subset_size = 1000  # Use 1000 samples for quick test
    
    X_train_small = X_train[:subset_size]
    y_train_small = y_train[:subset_size]
    X_val_small = X_val[:subset_size//4]  # 250 validation samples
    y_val_small = y_val[:subset_size//4]
    
    print(f"Training subset: {X_train_small.shape[0]} samples")
    print(f"Validation subset: {X_val_small.shape[0]} samples")
    
    # Define network architecture
    input_size = X_train.shape[1]  # 784 features
    num_classes = len(np.unique(y_train))  # 10 classes
    architecture = [input_size, 64, 32, num_classes]  # Simple 2-hidden layer network
    
    print(f"\nNetwork Architecture: {architecture}")
    print(f"Input size: {input_size}")
    print(f"Hidden layers: {architecture[1:-1]}")
    print(f"Output size: {num_classes}")
    
    # Create and train neural network
    print(f"\nCreating neural network...")
    nn = NeuralNetwork(
        layer_sizes=architecture,
        learning_rate=0.01,
        epochs=20,  # Reduced epochs for faster testing
        batch_size=32
    )
    
    print(f"Network created successfully!")
    print(f"Total parameters: {sum(w.size + b.size for w, b in zip(nn.weights, nn.biases)):,}")
    
    # Train the network
    print(f"\nStarting training...")
    try:
        training_time = nn.fit(X_train_small, y_train_small, verbose=True)
        print(f"Training completed successfully in {training_time:.2f} seconds!")
        
        # Evaluate on training and validation sets
        print(f"\nEvaluating model...")
        train_accuracy = nn.score(X_train_small, y_train_small)
        val_accuracy = nn.score(X_val_small, y_val_small)
        test_accuracy = nn.score(X_test, y_test)  # Evaluate on full test set
        
        print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Plot training loss
        print(f"\nPlotting training loss...")
        nn.plot_training_loss()
        
        # Test some predictions
        print(f"\nTesting predictions on a few samples...")
        sample_indices = np.random.choice(len(X_val_small), 5, replace=False)
        for i, idx in enumerate(sample_indices):
            prediction = nn.predict(X_val_small[idx:idx+1])
            true_label = y_val_small[idx]
            print(f"Sample {i+1}: Predicted {prediction[0]}, True {true_label}")
        
        print(f"\n" + "=" * 60)
        print("TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Training time: {training_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"\nERROR during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def simple_optimize_hyperparameters(X_train, y_train, X_val, y_val, max_trials=10):
    """
    Simple hyperparameter optimization without file saving
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        max_trials: Maximum number of trials
    
    Returns:
        best_config: Best hyperparameter configuration
        results: List of all trial results
    """
    print("=" * 60)
    print("SIMPLE HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Define hyperparameter search space
    input_size = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    # Simple architectures to try
    architectures = [
        [input_size, 64, 32, num_classes],           # Small network
        [input_size, 128, 64, num_classes],          # Medium network
        [input_size, 256, 128, 64, num_classes],     # Large network
        [input_size, 128, 128, num_classes],         # Wide network
    ]
    
    # Learning rates to try
    learning_rates = [0.001, 0.01, 0.1]
    
    # Epochs to try
    epochs_list = [20, 30, 50]
    
    # Batch sizes to try
    batch_sizes = [16, 32, 64]
    
    # Generate combinations
    combinations = []
    
    # Test each architecture with different learning rates
    for arch in architectures:
        for lr in learning_rates:
            for epochs in epochs_list:
                for batch_size in batch_sizes:
                    if len(combinations) < max_trials:
                        combinations.append({
                            'architecture': arch,
                            'learning_rate': lr,
                            'epochs': epochs,
                            'batch_size': batch_size
                        })
    
    # Shuffle combinations to get variety
    import random
    random.shuffle(combinations)
    combinations = combinations[:max_trials]
    
    results = []
    best_accuracy = 0
    best_config = None
    
    print(f"Testing {len(combinations)} combinations...")
    print(f"Architectures: {len(architectures)}")
    print(f"Learning rates: {learning_rates}")
    print(f"Epochs: {epochs_list}")
    print(f"Batch sizes: {batch_sizes}")
    
    for trial_count, combo in enumerate(combinations, 1):
        print(f"\n--- Trial {trial_count}/{len(combinations)} ---")
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
            
            # Evaluate on validation set
            val_accuracy = nn.score(X_val, y_val)
            
            # Store results
            result = {
                'trial': trial_count,
                'architecture': combo['architecture'],
                'learning_rate': combo['learning_rate'],
                'epochs': combo['epochs'],
                'batch_size': combo['batch_size'],
                'val_accuracy': val_accuracy,
                'training_time': training_time,
                'total_params': sum(w.size + b.size for w, b in zip(nn.weights, nn.biases))
            }
            
            results.append(result)
            
            print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
            print(f"Training Time: {training_time:.2f}s")
            print(f"Total Parameters: {result['total_params']:,}")
            
            # Update best configuration
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_config = result.copy()
                print("*** NEW BEST CONFIGURATION! ***")
            
        except Exception as e:
            print(f"Error in trial {trial_count}: {str(e)}")
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Total trials completed: {len(results)}")
    
    if best_config is None:
        print("ERROR: No successful trials completed.")
        return None, results
    
    print(f"Best validation accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print("\nBest configuration:")
    print(f"  Architecture: {best_config['architecture']}")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Epochs: {best_config['epochs']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  Training Time: {best_config['training_time']:.2f}s")
    print(f"  Total Parameters: {best_config['total_params']:,}")
    
    # Show top 3 configurations
    print(f"\nTop 3 configurations:")
    sorted_results = sorted(results, key=lambda x: x['val_accuracy'], reverse=True)
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"{i}. Accuracy: {result['val_accuracy']:.4f} - "
              f"Arch: {result['architecture']}, LR: {result['learning_rate']}, "
              f"Epochs: {result['epochs']}, Batch: {result['batch_size']}")
    
    return best_config, results

def run_simple_optimization():
    """Run simple hyperparameter optimization"""
    print("Starting simple hyperparameter optimization...")
    
    # Load MNIST data
    X_train, X_val, X_test, y_train, y_val, y_test = load_mnist_data()
    
    # Use smaller subset for faster optimization
    subset_size = 2000  # Use 2000 samples for optimization
    
    X_train_small = X_train[:subset_size]
    y_train_small = y_train[:subset_size]
    X_val_small = X_val[:subset_size//4]  # 500 validation samples
    y_val_small = y_val[:subset_size//4]
    
    print(f"Using {subset_size} training samples and {subset_size//4} validation samples for optimization")
    
    # Run optimization
    best_config, results = simple_optimize_hyperparameters(
        X_train_small, y_train_small, X_val_small, y_val_small, max_trials=10
    )
    
    if best_config is not None:
        print(f"\nOptimization completed!")
        print(f"Best configuration found:")
        print(f"  Architecture: {best_config['architecture']}")
        print(f"  Learning Rate: {best_config['learning_rate']}")
        print(f"  Epochs: {best_config['epochs']}")
        print(f"  Batch Size: {best_config['batch_size']}")
        print(f"  Best Validation Accuracy: {best_config['val_accuracy']:.4f}")
        
        # Train final model with best configuration on full dataset
        print(f"\nTraining final model with best configuration on full dataset...")
        final_nn = NeuralNetwork(
            layer_sizes=best_config['architecture'],
            learning_rate=best_config['learning_rate'],
            epochs=best_config['epochs'],
            batch_size=best_config['batch_size']
        )
        
        # Train on full dataset
        final_training_time = final_nn.fit(X_train, y_train, verbose=True)
        
        # Final evaluation
        final_train_accuracy = final_nn.score(X_train, y_train)
        final_val_accuracy = final_nn.score(X_val, y_val)
        final_test_accuracy = final_nn.score(X_test, y_test)
        
        print(f"\nFinal Results (Full Dataset):")
        print(f"Training Accuracy: {final_train_accuracy:.4f} ({final_train_accuracy*100:.2f}%)")
        print(f"Validation Accuracy: {final_val_accuracy:.4f} ({final_val_accuracy*100:.2f}%)")
        print(f"Test Accuracy: {final_test_accuracy:.4f} ({final_test_accuracy*100:.2f}%)")
        print(f"Training Time: {final_training_time:.2f}s")
        
        # Plot final training loss
        final_nn.plot_training_loss()
        
        return best_config, final_nn
    else:
        print("Optimization failed. No successful trials.")
        return None, None

def load_mb_data():
    """Load and preprocess MB (Mushroom Binary) dataset"""
        
    # Load training data
    print("Loading MB training data...")
    train_data = pd.read_csv("./MB_data_train.csv")
    print(f"MB training data shape: {train_data.shape}")
    
    # Warn about small dataset
    if len(train_data) <= 200:
        print("⚠️  WARNING: Very small dataset detected!")
        print("   - Using minimal network architectures")
        print("   - Using smaller learning rates")
        print("   - Using fewer epochs")
        print("   - Risk of overfitting is high")
    
    # Separate features and labels using 'y' column
    X_train_full = train_data.drop('y', axis=1).values  # All columns except 'y'
    y_train_full = train_data['y'].values               # 'y' column (labels)
    
    # Split training data into train and validation sets (80/20 split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    # For binary classification, we need to ensure labels are 0 and 1
    # Convert to binary if needed (assuming 0=edible, 1=poisonous)
    y_train = (y_train > 0).astype(int)
    y_val = (y_val > 0).astype(int)
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    return X_train, X_val, y_train, y_val

def load_mb_data_with_test():
    """Load MB data and create a synthetic test set"""
    print("Loading MB training data...")
    train_data = pd.read_csv("./MB_data_train.csv")
    print(f"MB training data shape: {train_data.shape}")
    
    # Warn about small dataset
    if len(train_data) <= 200:
        print("⚠️  WARNING: Very small dataset detected!")
        print("   - Using minimal network architectures")
        print("   - Using smaller learning rates")
        print("   - Using fewer epochs")
        print("   - Risk of overfitting is high")
        print("   - Creating synthetic test set from limited data")
    
    # Separate features and labels
    X_full = train_data.drop('y', axis=1).values
    y_full = train_data['y'].values
    
    # Convert to binary
    y_full = (y_full > 0).astype(int)
    
    # Create train/validation/test split (60/20/20)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    
    # Split remaining data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp  # 0.25 of 0.8 = 0.2 of total
    )
    
    print(f"Final split:")
    print(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    print(f"  Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"  Total samples: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}")
    print(f"  Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def simple_mb_test():
    """Simple test function to verify NeuralNetwork works on MB dataset"""
    print("=" * 60)
    print("SIMPLE MB (MUSHROOM BINARY) NEURAL NETWORK TEST")
    print("=" * 60)
    
    # Load MB data
    print("Loading MB dataset...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_mb_data_with_test()
    
    print(f"\nDataset Summary:")
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Validation set: {X_val.shape[0]} samples, {X_val.shape[1]} features")
    print(f"Number of classes: {len(np.unique(y_train))}")
    print(f"Class distribution: {np.bincount(y_train)}")
    
    # Use a smaller subset for faster testing (but use full dataset for very small datasets)
    if len(X_train) <= 200:  # Very small dataset
        print(f"\nUsing full dataset (very small dataset detected)")
        X_train_small = X_train
        y_train_small = y_train
        X_val_small = X_val
        y_val_small = y_val
    else:
        print(f"\nUsing smaller subset for faster testing...")
        subset_size = min(1000, len(X_train))  # Use up to 1000 samples for quick test
        
        X_train_small = X_train[:subset_size]
        y_train_small = y_train[:subset_size]
        X_val_small = X_val[:min(subset_size//4, len(X_val))]  # 250 validation samples or less
        y_val_small = y_val[:min(subset_size//4, len(X_val))]
    
    print(f"Training subset: {X_train_small.shape[0]} samples")
    print(f"Validation subset: {X_val_small.shape[0]} samples")
    
    # Define network architecture for binary classification
    input_size = X_train.shape[1]  # Number of features
    num_classes = 2  # Binary classification (edible/poisonous)
    
    # For very small dataset (100 samples), use very simple architecture
    if len(X_train) <= 200:  # Small dataset
        architecture = [input_size, 8, num_classes]  # Single hidden layer with only 8 neurons
        print("Using minimal architecture for small dataset")
    else:
        architecture = [input_size, 32, 16, num_classes]  # Simple 2-hidden layer network
    
    print(f"\nNetwork Architecture: {architecture}")
    print(f"Input size: {input_size}")
    print(f"Hidden layers: {architecture[1:-1]}")
    print(f"Output size: {num_classes} (binary classification)")
    
    # Create and train neural network with regularization
    print(f"\nCreating neural network...")
    nn = NeuralNetwork(
        layer_sizes=architecture,
        learning_rate=0.01,
        epochs=20,  # Reduced epochs for faster testing
        batch_size=min(16, len(X_train_small))  # Smaller batch size for small dataset
    )
    
    print(f"Network created successfully!")
    print(f"Total parameters: {sum(w.size + b.size for w, b in zip(nn.weights, nn.biases)):,}")
    
    # Train the network
    print(f"\nStarting training...")
    try:
        training_time = nn.fit(X_train_small, y_train_small, verbose=True)
        print(f"Training completed successfully in {training_time:.2f} seconds!")
        
        # Evaluate on training and validation sets
        print(f"\nEvaluating model...")
        train_accuracy = nn.score(X_train_small, y_train_small)
        val_accuracy = nn.score(X_val_small, y_val_small)
        test_accuracy = nn.score(X_test, y_test)  # Evaluate on test set
        
        print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Plot training loss
        print(f"\nPlotting training loss...")
        nn.plot_training_loss()
        
        # Test some predictions
        print(f"\nTesting predictions on a few samples...")
        sample_indices = np.random.choice(len(X_val_small), min(5, len(X_val_small)), replace=False)
        for i, idx in enumerate(sample_indices):
            prediction = nn.predict(X_val_small[idx:idx+1])
            true_label = y_val_small[idx]
            prediction_text = "Poisonous" if prediction[0] == 1 else "Edible"
            true_text = "Poisonous" if true_label == 1 else "Edible"
            print(f"Sample {i+1}: Predicted {prediction_text}, True {true_text}")
        
        print(f"\n" + "=" * 60)
        print("MB TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        print(f"Training time: {training_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"\nERROR during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def simple_mb_optimize_hyperparameters(X_train, y_train, X_val, y_val, max_trials=10):
    """
    Simple hyperparameter optimization for MB dataset without file saving
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        max_trials: Maximum number of trials
    
    Returns:
        best_config: Best hyperparameter configuration
        results: List of all trial results
    """
    print("=" * 60)
    print("SIMPLE MB HYPERPARAMETER OPTIMIZATION")
    print("=" * 60)
    
    # Define hyperparameter search space
    input_size = X_train.shape[1]
    num_classes = 2  # Binary classification
    
    # Simple architectures to try (much smaller for very small dataset)
    if len(X_train) <= 200:  # Small dataset
        architectures = [
            [input_size, 4, num_classes],               # Minimal network
            [input_size, 8, num_classes],               # Small network
            [input_size, 16, num_classes],              # Medium network
        ]
        print("Using minimal architectures for small dataset")
    else:
        architectures = [
            [input_size, 16, num_classes],              # Small network
            [input_size, 32, 16, num_classes],          # Medium network
            [input_size, 64, 32, num_classes],          # Large network
            [input_size, 32, 32, num_classes],          # Wide network
        ]
    
    # Learning rates to try (smaller for small dataset)
    if len(X_train) <= 200:
        learning_rates = [0.001, 0.005, 0.01]  # Smaller learning rates
    else:
        learning_rates = [0.001, 0.01, 0.1]
    
    # Epochs to try (fewer for small dataset)
    if len(X_train) <= 200:
        epochs_list = [10, 15, 20]  # Fewer epochs
    else:
        epochs_list = [20, 30, 50]
    
    # Batch sizes to try (smaller for small dataset)
    if len(X_train) <= 200:
        batch_sizes = [8, 16]  # Smaller batch sizes
    else:
        batch_sizes = [16, 32, 64]
    
    # Generate combinations
    combinations = []
    
    # Test each architecture with different learning rates
    for arch in architectures:
        for lr in learning_rates:
            for epochs in epochs_list:
                for batch_size in batch_sizes:
                    if len(combinations) < max_trials:
                        combinations.append({
                            'architecture': arch,
                            'learning_rate': lr,
                            'epochs': epochs,
                            'batch_size': batch_size
                        })
    
    # Shuffle combinations to get variety
    import random
    random.shuffle(combinations)
    combinations = combinations[:max_trials]
    
    results = []
    best_accuracy = 0
    best_config = None
    
    print(f"Testing {len(combinations)} combinations...")
    print(f"Architectures: {len(architectures)}")
    print(f"Learning rates: {learning_rates}")
    print(f"Epochs: {epochs_list}")
    print(f"Batch sizes: {batch_sizes}")
    
    for trial_count, combo in enumerate(combinations, 1):
        print(f"\n--- Trial {trial_count}/{len(combinations)} ---")
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
            
            # Evaluate on validation set
            val_accuracy = nn.score(X_val, y_val)
            
            # Store results
            result = {
                'trial': trial_count,
                'architecture': combo['architecture'],
                'learning_rate': combo['learning_rate'],
                'epochs': combo['epochs'],
                'batch_size': combo['batch_size'],
                'val_accuracy': val_accuracy,
                'training_time': training_time,
                'total_params': sum(w.size + b.size for w, b in zip(nn.weights, nn.biases))
            }
            
            results.append(result)
            
            print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
            print(f"Training Time: {training_time:.2f}s")
            print(f"Total Parameters: {result['total_params']:,}")
            
            # Update best configuration
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_config = result.copy()
                print("*** NEW BEST CONFIGURATION! ***")
            
        except Exception as e:
            print(f"Error in trial {trial_count}: {str(e)}")
            continue
    
    # Print summary
    print("\n" + "=" * 60)
    print("MB OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Total trials completed: {len(results)}")
    
    if best_config is None:
        print("ERROR: No successful trials completed.")
        return None, results
    
    print(f"Best validation accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print("\nBest configuration:")
    print(f"  Architecture: {best_config['architecture']}")
    print(f"  Learning Rate: {best_config['learning_rate']}")
    print(f"  Epochs: {best_config['epochs']}")
    print(f"  Batch Size: {best_config['batch_size']}")
    print(f"  Training Time: {best_config['training_time']:.2f}s")
    print(f"  Total Parameters: {best_config['total_params']:,}")
    
    # Show top 3 configurations
    print(f"\nTop 3 configurations:")
    sorted_results = sorted(results, key=lambda x: x['val_accuracy'], reverse=True)
    for i, result in enumerate(sorted_results[:3], 1):
        print(f"{i}. Accuracy: {result['val_accuracy']:.4f} - "
              f"Arch: {result['architecture']}, LR: {result['learning_rate']}, "
              f"Epochs: {result['epochs']}, Batch: {result['batch_size']}")
    
    return best_config, results

def run_simple_mb_optimization():
    """Run simple hyperparameter optimization for MB dataset"""
    print("Starting simple MB hyperparameter optimization...")
    
    # Load MB data
    X_train, X_val, X_test, y_train, y_val, y_test = load_mb_data_with_test()
    
    # Use smaller subset for faster optimization (but use full dataset for very small datasets)
    if len(X_train) <= 200:  # Very small dataset
        print(f"Using full dataset for optimization (very small dataset)")
        X_train_small = X_train
        y_train_small = y_train
        X_val_small = X_val
        y_val_small = y_val
    else:
        subset_size = min(2000, len(X_train))  # Use up to 2000 samples for optimization
        
        X_train_small = X_train[:subset_size]
        y_train_small = y_train[:subset_size]
        X_val_small = X_val[:min(subset_size//4, len(X_val))]  # 500 validation samples or less
        y_val_small = y_val[:min(subset_size//4, len(X_val))]
    
    print(f"Using {len(X_train_small)} training samples and {len(X_val_small)} validation samples for optimization")
    
    # Run optimization
    best_config, results = simple_mb_optimize_hyperparameters(
        X_train_small, y_train_small, X_val_small, y_val_small, max_trials=10
    )
    
    if best_config is not None:
        print(f"\nOptimization completed!")
        print(f"Best configuration found:")
        print(f"  Architecture: {best_config['architecture']}")
        print(f"  Learning Rate: {best_config['learning_rate']}")
        print(f"  Epochs: {best_config['epochs']}")
        print(f"  Batch Size: {best_config['batch_size']}")
        print(f"  Best Validation Accuracy: {best_config['val_accuracy']:.4f}")
        
        # Train final model with best configuration on full dataset
        print(f"\nTraining final model with best configuration on full dataset...")
        final_nn = NeuralNetwork(
            layer_sizes=best_config['architecture'],
            learning_rate=best_config['learning_rate'],
            epochs=best_config['epochs'],
            batch_size=best_config['batch_size']
        )
        
        # Train on full dataset
        final_training_time = final_nn.fit(X_train, y_train, verbose=True)
        
        # Final evaluation
        final_train_accuracy = final_nn.score(X_train, y_train)
        final_val_accuracy = final_nn.score(X_val, y_val)
        final_test_accuracy = final_nn.score(X_test, y_test)  # Evaluate on test set
        
        print(f"\nFinal Results (Full Dataset):")
        print(f"Training Accuracy: {final_train_accuracy:.4f} ({final_train_accuracy*100:.2f}%)")
        print(f"Validation Accuracy: {final_val_accuracy:.4f} ({final_val_accuracy*100:.2f}%)")
        print(f"Test Accuracy: {final_test_accuracy:.4f} ({final_test_accuracy*100:.2f}%)")
        print(f"Training Time: {final_training_time:.2f}s")
        
        # Plot final training loss
        final_nn.plot_training_loss()
        
        return best_config, final_nn
    else:
        print("Optimization failed. No successful trials.")
        return None, None

def evaluate_network_on_test_set(nn, X_test, y_test, dataset_name="Dataset"):
    """
    Evaluate a trained neural network on a test set
    
    Args:
        nn: Trained NeuralNetwork object
        X_test: Test features
        y_test: Test labels
        dataset_name: Name of the dataset for printing
    
    Returns:
        test_accuracy: Test accuracy score
    """
    print(f"\n" + "=" * 60)
    print(f"TEST SET EVALUATION - {dataset_name.upper()}")
    print("=" * 60)
    
    # Evaluate on test set
    test_accuracy = nn.score(X_test, y_test)
    
    print(f"Test Set Results:")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"  Class distribution: {np.bincount(y_test)}")
    
    # Test some predictions
    print(f"\nSample predictions:")
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        prediction = nn.predict(X_test[idx:idx+1])
        true_label = y_test[idx]
        
        if len(np.unique(y_test)) == 2:  # Binary classification
            prediction_text = "Class 1" if prediction[0] == 1 else "Class 0"
            true_text = "Class 1" if true_label == 1 else "Class 0"
        else:  # Multi-class
            prediction_text = f"Class {prediction[0]}"
            true_text = f"Class {true_label}"
        
        correct = "✓" if prediction[0] == true_label else "✗"
        print(f"  Sample {i+1}: Predicted {prediction_text}, True {true_text} {correct}")
    
    # Calculate additional metrics for binary classification
    if len(np.unique(y_test)) == 2:
        from sklearn.metrics import classification_report, confusion_matrix
        
        predictions = nn.predict(X_test)
        
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, predictions, target_names=['Class 0', 'Class 1']))
        
        print(f"Confusion Matrix:")
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        
        # Calculate precision, recall, F1-score manually
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nBinary Classification Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    print(f"\n" + "=" * 60)
    print(f"TEST EVALUATION COMPLETED")
    print("=" * 60)
    
    return test_accuracy

def comprehensive_mb_evaluation():
    """
    Comprehensive evaluation of MB dataset with train/validation/test split
    """
    print("=" * 60)
    print("COMPREHENSIVE MB DATASET EVALUATION")
    print("=" * 60)
    
    # Load data with test set
    X_train, X_val, X_test, y_train, y_val, y_test = load_mb_data_with_test()
    
    # Use full dataset for small datasets
    if len(X_train) <= 200:
        print(f"\nUsing full dataset (very small dataset detected)")
        X_train_use = X_train
        y_train_use = y_train
        X_val_use = X_val
        y_val_use = y_val
    else:
        # Use subset for larger datasets
        subset_size = min(1000, len(X_train))
        X_train_use = X_train[:subset_size]
        y_train_use = y_train[:subset_size]
        X_val_use = X_val[:min(subset_size//4, len(X_val))]
        y_val_use = y_val[:min(subset_size//4, len(X_val))]
    
    # Define architecture based on dataset size
    input_size = X_train.shape[1]
    num_classes = 2
    
    if len(X_train) <= 200:
        architecture = [input_size, 8, num_classes]
        epochs = 15
        learning_rate = 0.01
        batch_size = min(16, len(X_train_use))
    else:
        architecture = [input_size, 32, 16, num_classes]
        epochs = 20
        learning_rate = 0.01
        batch_size = 32
    
    print(f"\nNetwork Configuration:")
    print(f"  Architecture: {architecture}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    
    # Train model
    print(f"\nTraining model...")
    nn = NeuralNetwork(
        layer_sizes=architecture,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size
    )
    
    training_time = nn.fit(X_train_use, y_train_use, verbose=True)
    
    # Evaluate on all sets
    print(f"\nEvaluating model on all datasets...")
    train_accuracy = nn.score(X_train_use, y_train_use)
    val_accuracy = nn.score(X_val_use, y_val_use)
    
    print(f"\nTraining Results:")
    print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"  Training Time: {training_time:.2f}s")
    
    # Plot training loss
    nn.plot_training_loss()
    
    # Evaluate on test set
    test_accuracy = evaluate_network_on_test_set(nn, X_test, y_test, "MB Dataset")
    
    # Summary
    print(f"\nFINAL SUMMARY:")
    print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"  Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Check for overfitting
    overfitting_gap = train_accuracy - test_accuracy
    if overfitting_gap > 0.1:
        print(f"  ⚠️  Potential overfitting detected (gap: {overfitting_gap:.3f})")
    else:
        print(f"  ✅ Good generalization (gap: {overfitting_gap:.3f})")
    
    return nn, test_accuracy



if __name__ == "__main__":
    simple_mnist_test() 