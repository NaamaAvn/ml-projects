import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
from question_3 import SVM

# Ignore all warnings
warnings.filterwarnings('ignore')


def load_and_preprocess_data(csv_file, test_size=0.2, random_state=42):
    """
    Load and preprocess the breast cancer dataset.
    
    Args:
        csv_file (str): Path to the CSV file
        test_size (float): Proportion of data to use for testing
        random_state (int): Random seed for reproducibility
    
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    df = pd.read_csv(csv_file)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # Convert labels from {0, 1} to {-1, 1} for SVM
    y = np.where(y == 0, -1, 1)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Scale features since SVM is sensitive to feature scales
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def test_svm_models(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Test different SVM kernels and parameters.
    
    Args:
        X_train_scaled: Scaled training features
        X_test_scaled: Scaled test features
        y_train: Training labels
        y_test: Test labels
    
    Returns:
        tuple: (model_names, roc_data) where roc_data contains (fpr, tpr, auc_score) for each model
    """
    model_names = []
    roc_data = []  # Will store (fpr, tpr, auc_score) for each model
    
    print("Testing different SVM kernels and parameters:")
    print("=" * 50)
    
    for kernel in ['polynomial', 'rbf']:
        print(f"\nTesting {kernel} kernel:")
        print("-" * 30)
        
        if kernel == 'polynomial':
            for degree in [2, 3, 4, 5]:  # Reduced range for faster testing
                try:
                    svm = SVM(kernel=kernel, degree=degree, C=1.0)
                    svm.fit(X_train_scaled, y_train)
                    accuracy = svm.score(X_test_scaled, y_test)
                    
                    # Get decision function scores for ROC curve
                    decision_scores = svm.decision_function(X_test_scaled)
                    
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(y_test, decision_scores)
                    roc_auc = auc(fpr, tpr)
                    
                    # Store data
                    model_name = f"Polynomial (degree={degree})"
                    model_names.append(model_name)
                    roc_data.append((fpr, tpr, roc_auc))
                    
                    print(f"Degree: {degree}, Accuracy: {accuracy:.4f}, AUC: {roc_auc:.4f}")
                except Exception as e:
                    print(f"Degree: {degree}, Error: {str(e)}")
                    
        elif kernel == 'rbf':
            for gamma in [0.1, 0.5, 1.0, 2.0]:  # Reduced range for faster testing
                try:
                    svm = SVM(kernel=kernel, gamma=gamma, C=1.0)
                    svm.fit(X_train_scaled, y_train)
                    accuracy = svm.score(X_test_scaled, y_test)
                    
                    # Get decision function scores for ROC curve
                    decision_scores = svm.decision_function(X_test_scaled)
                    
                    # Calculate ROC curve
                    fpr, tpr, _ = roc_curve(y_test, decision_scores)
                    roc_auc = auc(fpr, tpr)
                    
                    # Store data
                    model_name = f"RBF (gamma={gamma})"
                    model_names.append(model_name)
                    roc_data.append((fpr, tpr, roc_auc))
                    
                    print(f"Gamma: {gamma}, Accuracy: {accuracy:.4f}, AUC: {roc_auc:.4f}")
                except Exception as e:
                    print(f"Gamma: {gamma}, Error: {str(e)}")
    
    return model_names, roc_data


def plot_roc_curves(model_names, roc_data, plot_filename='question_4_roc_curve.png'):
    """
    Plot ROC curves for all tested models.
    
    Args:
        model_names (list): List of model names
        roc_data (list): List of (fpr, tpr, auc_score) tuples
    """
    if not roc_data:
        print("No model results to plot.")
        return
    
    plt.figure(figsize=(10, 8))
    
    # Define colors and line styles for different models
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'olive']
    line_styles = ['-', '--', '-.', ':', '-', '--', '-', '--']
    
    # Plot ROC curves for each model
    for i, (name, (fpr, tpr, roc_auc)) in enumerate(zip(model_names, roc_data)):
        plt.plot(fpr, tpr, 
                color=colors[i % len(colors)],
                linestyle=line_styles[i % len(line_styles)],
                linewidth=2,
                label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Plot the diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.5, label='Random Classifier (AUC = 0.500)')
    
    # Customize the plot
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curves for Different SVM Configurations', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_filename)


def print_summary_statistics(model_names, roc_data):
    """
    Print summary statistics for all tested models.
    
    Args:
        model_names (list): List of model names
        roc_data (list): List of (fpr, tpr, auc_score) tuples
    """
    if not roc_data:
        print("No model results to summarize.")
        return
    
    auc_scores = [data[2] for data in roc_data]
    
    print("\n" + "=" * 60)
    print("SUMMARY OF ROC ANALYSIS:")
    print("=" * 60)
    for i, (name, (fpr, tpr, auc_score)) in enumerate(zip(model_names, roc_data)):
        print(f"{name:<25} | AUC: {auc_score:.4f}")
    
    best_idx = np.argmax(auc_scores)
    print(f"\nBest performing model: {model_names[best_idx]} (AUC = {auc_scores[best_idx]:.4f})")


def main():
    """
    Main function to orchestrate the SVM analysis workflow.
    """
    # Load and preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = load_and_preprocess_data(
        "Processed Wisconsin Diagnostic Breast Cancer.csv"
    )
    
    # Test different SVM models
    model_names, roc_data = test_svm_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Create visualizations
    plot_roc_curves(model_names, roc_data)
    
    # Print summary statistics
    print_summary_statistics(model_names, roc_data)


if __name__ == "__main__":
    main()
                    