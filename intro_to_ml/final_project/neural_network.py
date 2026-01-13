import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import time

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