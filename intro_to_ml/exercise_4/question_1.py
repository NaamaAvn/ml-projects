import pandas as pd
import numpy as np
import qpsolvers as qps
import matplotlib.pyplot as plt
import matplotlib
import warnings

# Ignore all warnings
warnings.filterwarnings('ignore')

def svm_primal(X, y, max_iter=4000, verbose=False):
    
    N = X.shape[0]
    n = X.shape[1]
    P = np.eye(n)
    q = np.zeros(n)
    G = -np.diag(y) @ X
    h = -np.ones(N)

    w = qps.solve_qp(P, q, G, h, solver='osqp', max_iter=max_iter, verbose=verbose)
    
    return w

def svm_dual(X, y, max_iter=4000, verbose=False):
    
    N = X.shape[0]
    G = np.diag(y) @ X
    P = 0.5 * G @ G.T
    q = -np.ones(N)
    GG = -np.eye(N)
    h = np.zeros(N)

    alpha = qps.solve_qp(P, q, GG, h, solver='osqp', max_iter=max_iter, verbose=verbose)
    
    w = G.T @ alpha
    
    return alpha, 0.5*w

def my_plot_classifier(w, X, y, plot_filename):
    # Create a new figure to avoid duplicate legend entries
    plt.figure()
    # Plot the data with the linear separator
    plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Class -1')
    plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Class +1')

    # Plot the decision boundary: w[0] * x1 + w[1] * x2 + w[2] = 0
    x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    y_vals = -(w[0] * x_vals + w[2]) / w[1]
    plt.plot(x_vals, y_vals, color='black', label='Decision boundary')

    # Compute margins to find support vectors
    margins = y * (X @ w)
    support_vector_indices = np.where(np.abs(margins - 1.0) < 1e-2)[0]
    print(f"Support vector indices: {support_vector_indices}")
    support_vectors = X[support_vector_indices]

    # Then plot hollow circles around the support vector points
    plt.scatter(support_vectors[:, 0], support_vectors[:, 1], 
                facecolors='none', s=200, marker='o',
                edgecolors='black', linewidths=1.5)

    # Plot margin lines
    margin = 1 / np.linalg.norm(w[:2])

    # y = -(w0 * x + w2 ± 1) / w1
    y_vals_margin_pos = -(w[0] * x_vals + w[2] - 1) / w[1]
    y_vals_margin_neg = -(w[0] * x_vals + w[2] + 1) / w[1]

    plt.plot(x_vals, y_vals_margin_pos, 'k--', label='Positive Margin (+1)')
    plt.plot(x_vals, y_vals_margin_neg, 'k--', label='Negative Margin (-1)')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title(f"{plot_filename} - Linear Classifier")
    plt.grid(True)
    plt.savefig(plot_filename)

def main_question_1_a(X, y):
    w = svm_primal(X, y)

    print("QP solution: w = {}".format(w))

    my_plot_classifier(w, X, y, "Question_1a_Primal_SVM.png")


def main_question_1_b(X, y):
    alpha, w = svm_dual(X, y)

    print("Translating to the dual we get: w_dual =", w)

    my_plot_classifier(w, X, y, "Question_1b_Dual_SVM.png")

if __name__ == "__main__":

    # Load data simple_classification.csv
    df = pd.read_csv("simple_classification.csv")
    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    N = X.shape[0]
    X = np.c_[X, np.ones(N)]
    y = y * 2 - 1  # Convert {0,1} to {-1,1}
    
    # run main_question_1_a()
    main_question_1_a(X, y)

    # run main_question_1_b()
    main_question_1_b(X, y)
