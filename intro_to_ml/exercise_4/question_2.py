import pandas as pd
import numpy as np
import qpsolvers as qps
import matplotlib.pyplot as plt
import matplotlib
import itertools
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

def plot_data(X, y, zoom_out=False, s=None):
    
    if zoom_out:
        x_min = np.amin(X[:, 0])
        x_max = np.amax(X[:, 0])
        y_min = np.amin(X[:, 1])
        y_max = np.amax(X[:, 1])

        plt.axis([x_min-1, x_max+1, y_min-1, y_max+1])

    plt.scatter(X[:,0], X[:,1], c=y, s=s, cmap=matplotlib.colors.ListedColormap(['blue','red']))

def svm_dual_kernel(X, y, ker, max_iter=4000, verbose=False):
    
    N = X.shape[0]
    P = np.empty((N, N))
    for i, j in itertools.product(range(N), range(N)):
        P[i, j] = y[i] * y[j] * ker(X[i,:], X[j,:])
    P = 0.5*(P+P.T)
    P = 0.5*P
    q = -np.ones(N)
    GG = -np.eye(N)
    h = np.zeros(N)

    alpha = qps.solve_qp(P, q, GG, h, solver='osqp', max_iter=max_iter, verbose=verbose)
    # w = \sum_i alpha_iy_ix_i
    # w = G.T @ alpha
    
    return alpha

def plot_classifier_z_kernel(alpha, X, y, ker, s=None, kernel_name="kernel", param_value=None):
    x_min = np.amin(X[:, 0])
    x_max = np.amax(X[:, 0])
    y_min = np.amin(X[:, 1])
    y_max = np.amax(X[:, 1])
    
    xx = np.linspace(x_min, x_max)
    yy = np.linspace(y_min, y_max)
    
    xx, yy = np.meshgrid(xx, yy)
    
    N = X.shape[0]
    z = np.zeros(xx.shape)
    for i, j in itertools.product(range(xx.shape[0]), range(xx.shape[1])):
        z[i, j] = sum([y[k]*alpha[k]*ker(X[k,:], np.array([xx[i,j],yy[i,j]])) for k in range(N)])
    
    plt.rcParams["figure.figsize"] = [15, 10]
                            
    plt.contour(xx, yy, z, levels=[-1, 0, 1], colors=['blue', 'black', 'red'], linestyles=['--','-', '--'])
    
    # Plot regular points
    plot_data(X, y, s=s)
    
    # Identify and plot support vectors
    # Support vectors are points with non-zero alpha values (using a small threshold)
    threshold = 0.01
    support_vectors = np.abs(alpha) > threshold
    
    
    # Then plot hollow circles around the support vector points
    plt.scatter(X[support_vectors, 0], X[support_vectors, 1], 
                facecolors='none', s=200, marker='o',
                edgecolors='black', linewidths=1.5)
    
    # Create filename based on kernel type and parameter
    if param_value is not None:
        filename = f"question_2_svm_{kernel_name}_{param_value}.png"
    else:
        filename = f"question_2_svm_{kernel_name}.png"
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

def polynomial_kernel(x, y, degree=2):
    
    return (1+x.T @ y)**degree

def rbf_kernel(x, y, gamma=1):

    return np.exp(-gamma * np.linalg.norm(x - y)**2)

def svm_polynomial_kernel(X, y, degree):
    # Create a kernel function with the specified degree
    def kernel_with_degree(x, y):
        return polynomial_kernel(x, y, degree=degree)

    alpha = svm_dual_kernel(X, y, kernel_with_degree)

    if alpha is None:
        print("No solution found")
        return

    plot_classifier_z_kernel(alpha, X, y, kernel_with_degree, kernel_name="polynomial", param_value=degree)

def svm_rbf_kernel(X, y, gamma):
    def kernel_with_gamma(x, y):
        return rbf_kernel(x, y, gamma=gamma)

    alpha = svm_dual_kernel(X, y, kernel_with_gamma)

    if alpha is None:
        print("No solution found")
        return

    plot_classifier_z_kernel(alpha, X, y, kernel_with_gamma, kernel_name="rbf", param_value=gamma)

def predict_proba(X_train, X_test, y_train, alpha, ker):
    """Predict probabilities for test data using the trained SVM"""
    N_train = X_train.shape[0]
    N_test = X_test.shape[0]
    scores = np.zeros(N_test)
    
    for i in range(N_test):
        scores[i] = sum([y_train[k]*alpha[k]*ker(X_train[k,:], X_test[i,:]) 
                        for k in range(N_train)])
    return scores

def evaluate_predictions(y_true, y_pred):
    """Calculate accuracy of predictions"""
    return np.mean(y_true == y_pred)

def plot_roc_curve(poly_results, rbf_results, y_test):
    """Plot ROC curves for the different kernel configurations"""
    from sklearn.metrics import roc_curve, auc
    
    # Create new figure
    plt.figure(figsize=(10, 8))
    
    # Plot random classifier baseline
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    # Plot polynomial kernel ROC curves
    for degree, data in poly_results.items():
        if 'test_scores' in data:
            fpr, tpr, _ = roc_curve(y_test, data['test_scores'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, 
                    label=f'Polynomial deg={degree} (AUC = {roc_auc:.2f})')
    
    # Plot RBF kernel ROC curves  
    for gamma, data in rbf_results.items():
        if 'test_scores' in data:
            fpr, tpr, _ = roc_curve(y_test, data['test_scores'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2,
                    label=f'RBF gamma={gamma} (AUC = {roc_auc:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('question_2_roc_curve.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_performance_comparison(poly_results, rbf_results):
    """Create a bar plot comparing performance of different kernel parameters"""
    plt.figure(figsize=(12, 6))
    
    # Prepare data for plotting
    poly_degrees = [str(d) for d in poly_results.keys()]
    poly_accuracies = [results['test_acc'] for results in poly_results.values()]
    poly_errors = [1 - acc for acc in poly_accuracies]
    
    rbf_gammas = [str(g) for g in rbf_results.keys()]
    rbf_accuracies = [results['test_acc'] for results in rbf_results.values()]
    rbf_errors = [1 - acc for acc in rbf_accuracies]
    
    # Set up the plot
    x = np.arange(len(poly_degrees) + len(rbf_gammas))
    width = 0.35
    
    # Create bars
    plt.bar(x[:len(poly_degrees)], poly_accuracies, width, label='Polynomial Kernel Accuracy', color='skyblue')
    plt.bar(x[:len(poly_degrees)], poly_errors, width, bottom=poly_accuracies, label='Polynomial Kernel Error', color='lightcoral')
    
    plt.bar(x[len(poly_degrees):], rbf_accuracies, width, label='RBF Kernel Accuracy', color='royalblue')
    plt.bar(x[len(poly_degrees):], rbf_errors, width, bottom=rbf_accuracies, label='RBF Kernel Error', color='indianred')
    
    # Add labels and title
    plt.xlabel('Kernel Parameters')
    plt.ylabel('Performance')
    plt.title('Test Set Performance Comparison')
    plt.xticks(x, poly_degrees + rbf_gammas)
    plt.legend()
    
    # Add value labels on top of bars
    for i, v in enumerate(poly_accuracies + rbf_accuracies):
        plt.text(i, v + 0.01, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('question_2_performance_comparison.png', bbox_inches='tight', dpi=300)
    plt.close()

def main_question_2():
    # Load and split data
    df = pd.read_csv("simple_nonlin_classification.csv")
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    
    # Split data into train and test sets (80-20 split)
    np.random.seed(42)  # for reproducibility
    indices = np.random.permutation(len(X))
    train_size = int(0.8 * len(X))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    print("Dataset split into:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Store results for comparison
    poly_results = {}
    rbf_results = {}
    
    # Try different polynomial degrees
    print("\nPolynomial Kernel Results:")
    print("Degree | Train Accuracy | Test Accuracy")
    print("----------------------------------------")
    for degree in [2, 3, 4, 5, 6]:
        # Train
        def kernel_with_degree(x, y):
            return polynomial_kernel(x, y, degree=degree)
        
        alpha = svm_dual_kernel(X_train, y_train, kernel_with_degree)
        
        if alpha is None:
            print(f"{degree:6d} | No solution found")
            continue
            
        # Make predictions
        train_scores = predict_proba(X_train, X_train, y_train, alpha, kernel_with_degree)
        test_scores = predict_proba(X_train, X_test, y_train, alpha, kernel_with_degree)
        
        # Calculate accuracies
        train_acc = evaluate_predictions(y_train, np.sign(train_scores))
        test_acc = evaluate_predictions(y_test, np.sign(test_scores))
        
        # Store results
        poly_results[degree] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_scores': test_scores
        }
        
        print(f"{degree:6d} | {train_acc:13.3f} | {test_acc:12.3f}")
        
        # Plot the decision boundary
        plot_classifier_z_kernel(alpha, X_train, y_train, kernel_with_degree, 
                               kernel_name="polynomial", param_value=degree)

    # Try different gamma values for rbf kernel
    print("\nRBF Kernel Results:")
    print("Gamma | Train Accuracy | Test Accuracy")
    print("----------------------------------------")
    for gamma in [0.1, 0.5, 1, 2, 5]:
        # Train
        def kernel_with_gamma(x, y):
            return rbf_kernel(x, y, gamma=gamma)
        
        alpha = svm_dual_kernel(X_train, y_train, kernel_with_gamma)
        
        if alpha is None:
            print(f"{gamma:5.1f} | No solution found")
            continue
            
        # Make predictions
        train_scores = predict_proba(X_train, X_train, y_train, alpha, kernel_with_gamma)
        test_scores = predict_proba(X_train, X_test, y_train, alpha, kernel_with_gamma)
        
        # Calculate accuracies
        train_acc = evaluate_predictions(y_train, np.sign(train_scores))
        test_acc = evaluate_predictions(y_test, np.sign(test_scores))
        
        # Store results
        rbf_results[gamma] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'test_scores': test_scores
        }
        
        print(f"{gamma:5.1f} | {train_acc:13.3f} | {test_acc:12.3f}")
        
        # Plot the decision boundary
        plot_classifier_z_kernel(alpha, X_train, y_train, kernel_with_gamma, 
                               kernel_name="rbf", param_value=gamma)
    
    # Create performance comparison plot
    plot_performance_comparison(poly_results, rbf_results)
    # Plot ROC curves
    plot_roc_curve(poly_results, rbf_results, y_test)

if __name__ == "__main__":
    # Run question 2
    main_question_2()
