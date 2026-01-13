import numpy as np
import qpsolvers as qps
from sklearn.metrics import accuracy_score
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

class SVM():
    def __init__(self, kernel='rbf', degree=2, C=1.0, gamma=1.0, coef0=0.0):
        """
        Initialize the SVM classifier.
        
        Parameters:
        -----------
        kernel : str, default='rbf'
            The kernel type ('polynomial', 'rbf' or 'sigmoid')
        degree : int, default=2
            Degree of polynomial kernel (only used if kernel='polynomial')
        C : float, default=1.0
            Regularization parameter
        gamma : float, default=1.0
            Kernel coefficient 
        coef0 : float, default=0.0
            Independent term in kernel function (only used if kernel='sigmoid')
        """
        self.kernel = kernel
        self.degree = degree
        self.C = C
        self.gamma = gamma
        self.coef0 = coef0
        self.alpha = None
        self.X_train = None
        self.y_train = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_alphas = None
        
    
    def _polynomial_kernel(self, x1, x2):
        """Polynomial kernel function"""
        return (1 + np.dot(x1, x2)) ** self.degree
    
    def _rbf_kernel(self, x1, x2):
        """RBF kernel function"""
        return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
    
    def _sigmoid_kernel(self, x1, x2):
        """Sigmoid kernel function"""
        return np.tanh(self.gamma * np.dot(x1, x2) + self.coef0)
    
    def _get_kernel_function(self):
        """Return the appropriate kernel function"""
        if self.kernel == 'polynomial':
            return self._polynomial_kernel
        elif self.kernel == 'rbf':
            return self._rbf_kernel
        elif self.kernel == 'sigmoid':
            return self._sigmoid_kernel
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X, y):
        """
        Fit the SVM model according to the given training data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training vectors
        y : array-like of shape (n_samples,)
            Target values (-1 or 1)
            
        Returns:
        --------
        self : object
            Returns self
        """
        n_samples, n_features = X.shape
        
        # Store training data
        self.X_train = X
        self.y_train = y
        
        # Get kernel function
        kernel_func = self._get_kernel_function()
        
        # Compute kernel matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = kernel_func(X[i], X[j])
        
        # Set up the quadratic programming problem
        P = np.outer(y, y) * K
        q = -np.ones(n_samples)
        G = np.vstack((-np.eye(n_samples), np.eye(n_samples)))
        h = np.hstack((np.zeros(n_samples), self.C * np.ones(n_samples)))
        A = y.reshape(1, -1)
        b = np.array([0.0])

        
        # Solve the quadratic programming problem
        self.alpha = qps.solve_qp(P, q, G, h, A, b, solver='osqp')
        
        # Check if solver returned a valid solution
        if self.alpha is None:
            raise ValueError("QP solver failed to find a solution. This might be due to numerical issues or infeasible problem. Try adjusting parameters or scaling the data.")
        
        # Find support vectors
        support_vector_mask = self.alpha > 1e-5
        self.support_vectors = X[support_vector_mask]
        self.support_vector_labels = y[support_vector_mask]
        self.support_vector_alphas = self.alpha[support_vector_mask]
        
        return self
    
    def decision_function(self, X):
        """
        Evaluate the decision function for the samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        X : ndarray of shape (n_samples,)
            The decision function of the samples
        """
        if self.alpha is None:
            raise ValueError("Model has not been fitted yet.")
            
        kernel_func = self._get_kernel_function()
        n_samples = X.shape[0]
        decisions = np.zeros(n_samples)
        
        for i in range(n_samples):
            decisions[i] = np.sum(
                self.support_vector_alphas * 
                self.support_vector_labels * 
                np.array([kernel_func(x, X[i]) for x in self.support_vectors])
            )
            
        return decisions
    
    def predict(self, X):
        """
        Perform classification on samples in X.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Samples
            
        Returns:
        --------
        y_pred : ndarray of shape (n_samples,)
            Class labels for samples in X
        """
        decisions = self.decision_function(X)
        return np.sign(decisions)
    
    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True labels for X
            
        Returns:
        --------
        score : float
            Mean accuracy of self.predict(X) wrt. y
        """
        return accuracy_score(y, self.predict(X))

