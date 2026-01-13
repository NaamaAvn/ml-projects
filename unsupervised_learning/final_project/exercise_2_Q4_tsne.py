import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from Q2 import generate_swiss_roll, generate_sinusoidal_curve

def compute_high_dim_similarities(X, perplexity=30.0):
    """Compute pairwise conditional probabilities p_{j|i} using a Gaussian kernel with fixed perplexity."""
    (n, d) = X.shape
    D = pairwise_distances(X, squared=True)
    P = np.zeros((n, n))
    logU = np.log(perplexity)

    for i in range(n):
        beta = 1.0
        betamin = -np.inf
        betamax = np.inf
        Di = np.delete(D[i], i)
        H, thisP = compute_entropy(Di, beta)
        H_diff = H - logU
        tries = 0

        while np.abs(H_diff) > 1e-5 and tries < 50:
            if H_diff > 0:
                betamin = beta
                beta = 2.0 * beta if betamax == np.inf else (beta + betamax) / 2.
            else:
                betamax = beta
                beta = beta / 2.0 if betamin == -np.inf else (beta + betamin) / 2.

            H, thisP = compute_entropy(Di, beta)
            H_diff = H - logU
            tries += 1

        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    return P

def compute_entropy(D, beta):
    P = np.exp(-D * beta)
    sumP = np.sum(P)
    
    # Avoid numerical issues
    if sumP < 1e-20:
        return 0.0, np.ones_like(P) / len(P)
    
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    return H, P / sumP

def compute_low_dim_similarities(Y):
    """Compute low-dimensional similarities using Gaussian kernel (not Student-t)"""
    D = pairwise_distances(Y, squared=True)
    # Use Gaussian kernel like in high dimensions
    Q = np.exp(-D)
    np.fill_diagonal(Q, 0.0)
    sum_Q = np.sum(Q)
    if sum_Q < 1e-20:
        sum_Q = 1e-20
    return Q / sum_Q

def compute_low_dim_similarities_tsne(Y):
    """Compute low-dimensional similarities using Student-t distribution with 1 degree of freedom"""
    D = pairwise_distances(Y, squared=True)
    # Student-t distribution with 1 degree of freedom: (1 + ||y_i - y_j||^2)^(-1)
    Q = 1.0 / (1.0 + D)
    np.fill_diagonal(Q, 0.0)
    sum_Q = np.sum(Q)
    if sum_Q < 1e-20:
        sum_Q = 1e-20
    return Q / sum_Q

def compute_low_dim_similarities_tsne_general(Y, degrees_of_freedom=1.0):
    """Compute low-dimensional similarities using Student-t distribution with arbitrary degrees of freedom"""
    D = pairwise_distances(Y, squared=True)  # ||y_i - y_j||²
    nu = degrees_of_freedom
    
    # General Student-t distribution: (1 + ||y_i - y_j||²/ν)^(-(ν+1)/2)
    Q = (1.0 + D / nu) ** (-(nu + 1.0) / 2.0)
    np.fill_diagonal(Q, 0.0)
    sum_Q = np.sum(Q)
    if sum_Q < 1e-20:
        sum_Q = 1e-20
    return Q / sum_Q

def sne(X, dim=2, perplexity=30.0, learning_rate=10.0, max_iter=1000):
    """Symmetric SNE implementation"""
    n, d = X.shape
    
    # Initialize Y with small random values
    Y = np.random.normal(0, 1e-4, (n, dim))

    # Compute high-dimensional similarities
    P = compute_high_dim_similarities(X, perplexity)
    P = (P + P.T) / (2.0 * n)  # Make symmetric and normalize
    P = np.maximum(P, 1e-12)   # Avoid zeros
    
    # Keep track of previous Y for momentum (optional)
    dY_prev = np.zeros_like(Y)
    
    for iter in range(max_iter):
        # Compute low-dimensional similarities
        Q = compute_low_dim_similarities(Y)
        Q = np.maximum(Q, 1e-12)  # Avoid zeros
        
        # Compute gradient using correct SNE formula
        dY = np.zeros_like(Y)
        
        for i in range(n):
            diff = Y[i] - Y  # Shape: (n, dim)
            # SNE gradient: sum over j of (P_ij - Q_ij) * (y_i - y_j)
            pq_diff = (P[i] - Q[i]).reshape(-1, 1)  # Shape: (n, 1)
            dY[i] = 2.0 * np.sum(pq_diff * diff, axis=0)
        
        # Check for numerical issues
        if np.any(np.isnan(dY)) or np.any(np.isinf(dY)):
            print(f"Warning: Numerical issues detected at iteration {iter}")
            break
        
        # Gradient clipping for stability
        grad_norm = np.linalg.norm(dY)
        if grad_norm > 100:
            dY = dY / grad_norm * 100
            
        # Update Y with momentum
        momentum = 0.5 if iter < 250 else 0.8
        dY = momentum * dY_prev + learning_rate * dY
        Y = Y - dY
        dY_prev = dY
        
        # Center Y around origin
        Y = Y - np.mean(Y, axis=0)

        if iter % 100 == 0:
            # Compute KL divergence safely
            kl_div = np.sum(P * np.log(np.clip(P / Q, 1e-12, 1e12)))
            print(f"Iteration {iter}, KL divergence: {kl_div:.4f}")

    return Y

def tsne(X, dim=2, perplexity=30.0, learning_rate=200.0, max_iter=1000):
    """t-SNE implementation using Student-t distribution with 1 degree of freedom"""
    n, d = X.shape
    
    # Initialize Y with small random values
    Y = np.random.normal(0, 1e-4, (n, dim))

    # Compute high-dimensional similarities (same as SNE)
    P = compute_high_dim_similarities(X, perplexity)
    P = (P + P.T) / (2.0 * n)  # Make symmetric and normalize
    P = np.maximum(P, 1e-12)   # Avoid zeros
    
    # Keep track of previous Y for momentum
    dY_prev = np.zeros_like(Y)
    
    for iter in range(max_iter):
        # Compute low-dimensional similarities using Student-t distribution
        Q = compute_low_dim_similarities_tsne(Y)
        Q = np.maximum(Q, 1e-12)  # Avoid zeros
        
        # Compute gradient using t-SNE formula
        dY = np.zeros_like(Y)
        
        for i in range(n):
            diff = Y[i] - Y  # Shape: (n, dim)
            # t-SNE gradient: 4 * sum over j of (P_ij - Q_ij) * (y_i - y_j) * (1 + ||y_i - y_j||^2)^(-1)
            squared_distances = np.sum(diff ** 2, axis=1)  # Shape: (n,)
            pq_diff = P[i] - Q[i]  # Shape: (n,)
            # The (1 + ||y_i - y_j||^2)^(-1) factor
            weights = 1.0 / (1.0 + squared_distances)  # Shape: (n,)
            # Combine everything
            dY[i] = 4.0 * np.sum((pq_diff * weights).reshape(-1, 1) * diff, axis=0)
        
        # Check for numerical issues
        if np.any(np.isnan(dY)) or np.any(np.isinf(dY)):
            print(f"Warning: Numerical issues detected at iteration {iter}")
            break
        
        # Gradient clipping for stability
        grad_norm = np.linalg.norm(dY)
        if grad_norm > 100:
            dY = dY / grad_norm * 100
            
        # Update Y with momentum (different schedule for t-SNE)
        momentum = 0.5 if iter < 250 else 0.8
        dY = momentum * dY_prev + learning_rate * dY
        Y = Y - dY
        dY_prev = dY
        
        # Center Y around origin
        Y = Y - np.mean(Y, axis=0)

        if iter % 100 == 0:
            # Compute KL divergence safely
            kl_div = np.sum(P * np.log(np.clip(P / Q, 1e-12, 1e12)))
            print(f"Iteration {iter}, KL divergence: {kl_div:.4f}")

    return Y

def tsne_general(X, degrees_of_freedom, dim=2, perplexity=30.0, learning_rate=200.0, max_iter=1000):
    """t-SNE implementation using Student-t distribution with arbitrary degrees of freedom"""
    n, d = X.shape
    nu = degrees_of_freedom
    
    # Initialize Y with small random values
    Y = np.random.normal(0, 1e-4, (n, dim))

    # Compute high-dimensional similarities (same as SNE)
    P = compute_high_dim_similarities(X, perplexity)
    P = (P + P.T) / (2.0 * n)  # Make symmetric and normalize
    P = np.maximum(P, 1e-12)   # Avoid zeros
    
    # Keep track of previous Y for momentum
    dY_prev = np.zeros_like(Y)
    
    for iter in range(max_iter):
        # Compute low-dimensional similarities using Student-t distribution
        Q = compute_low_dim_similarities_tsne_general(Y, nu)
        Q = np.maximum(Q, 1e-12)  # Avoid zeros
        
        # Compute gradient using generalized t-SNE formula
        dY = np.zeros_like(Y)
        
        for i in range(n):
            diff = Y[i] - Y  # Shape: (n, dim)
            squared_distances = np.sum(diff ** 2, axis=1)  # Shape: (n,)
            pq_diff = P[i] - Q[i]  # Shape: (n,)
            
            # Generalized t-SNE gradient factor: (1 + ||y_i - y_j||²/ν)^(-1)
            # This comes from the derivative of the Student-t distribution
            weights = (1.0 + squared_distances / nu) ** (-1.0)  # Shape: (n,)
            
            # The gradient factor 4 becomes 4(ν+1)/ν for general ν
            gradient_factor = 4.0 * (nu + 1.0) / nu
            
            # Combine everything
            dY[i] = gradient_factor * np.sum((pq_diff * weights).reshape(-1, 1) * diff, axis=0)
        
        # Check for numerical issues
        if np.any(np.isnan(dY)) or np.any(np.isinf(dY)):
            print(f"Warning: Numerical issues detected at iteration {iter}")
            break
        
        # Gradient clipping for stability
        grad_norm = np.linalg.norm(dY)
        if grad_norm > 100:
            dY = dY / grad_norm * 100
            
        # Update Y with momentum
        momentum = 0.5 if iter < 250 else 0.8
        dY = momentum * dY_prev + learning_rate * dY
        Y = Y - dY
        dY_prev = dY
        
        # Center Y around origin
        Y = Y - np.mean(Y, axis=0)

        if iter % 100 == 0:
            # Compute KL divergence safely
            kl_div = np.sum(P * np.log(np.clip(P / Q, 1e-12, 1e12)))
            print(f"Iteration {iter}, KL divergence: {kl_div:.4f}")

    return Y

def main_4_2():
    # swiss roll dataset
    X, t, assigned_colors = generate_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
    print(f"Swiss roll dataset shape: {X.shape}")
    print(f"Running t-SNE (ν=1)...")
    Y = tsne(X, dim=2, perplexity=40.0, learning_rate=200.0, max_iter=500)
    plt.scatter(Y[:, 0], Y[:, 1], c=assigned_colors, s=5)
    plt.title("t-SNE (ν=1) Embedding of Swiss Roll")
    plt.savefig("exercise_2/Q4_2_tsne_swiss_roll.png", dpi=300, bbox_inches='tight')
    
    # Sinusoidal curve dataset
    X_coords, Y_coords, Z_coords, t = generate_sinusoidal_curve()
    X_sinusoidal = np.column_stack([X_coords, Y_coords, Z_coords])
    print(f"Sinusoidal curve dataset shape: {X_sinusoidal.shape}")
    print(f"Running t-SNE (ν=1)...")
    Y = tsne(X_sinusoidal, dim=2, perplexity=40.0, learning_rate=200.0, max_iter=500)
    plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], c=t, s=5)
    plt.title("t-SNE (ν=1) Embedding of Sinusoidal Curve")
    plt.savefig("exercise_2/Q4_2_tsne_sinusoidal_curve.png", dpi=300, bbox_inches='tight')
    

def main_4_3():
    # D_1_2_4_7 dataset
    digits = load_digits()
    # filter out the digits that are not 1, 2, 4, 7
    mask = np.isin(digits.target, [1, 2, 4, 7])
    X = digits.data[mask]
    y = digits.target[mask]
    
    # Normalize the data
    X = X / 255.0  # Normalize pixel values to [0, 1]

    print(f"D_1_2_4_7 dataset shape: {X.shape}")
    print(f"Running SNE...")
    
    Y_sne = sne(X, perplexity=30.0, learning_rate=10.0, max_iter=500)
    
    print(f"Running t-SNE (ν=1)...")
    Y_tsne_1 = tsne(X, perplexity=30.0, learning_rate=200.0, max_iter=500)
    
    print(f"Running t-SNE (ν=3)...")
    Y_tsne_3 = tsne_general(X, degrees_of_freedom=3.0, perplexity=30.0, learning_rate=200.0, max_iter=500)
    
    print(f"Running t-SNE (ν=10)...")
    Y_tsne_10 = tsne_general(X, degrees_of_freedom=10.0, perplexity=30.0, learning_rate=200.0, max_iter=500)

    # Plot all embeddings
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['red', 'green', 'blue', 'orange']
    
    embeddings = [
        (Y_sne, "SNE Embedding"),
        (Y_tsne_1, "t-SNE (ν=1)"),
        (Y_tsne_3, "t-SNE (ν=3)"),
        (Y_tsne_10, "t-SNE (ν=10)")
    ]
    
    for idx, (Y_embed, title) in enumerate(embeddings):
        ax = axes[idx // 2, idx % 2]
        
        for i, digit in enumerate([1, 2, 4, 7]):
            mask_digit = y == digit
            ax.scatter(Y_embed[mask_digit, 0], Y_embed[mask_digit, 1], 
                      c=colors[i], label=f'Digit {digit}', s=20, alpha=0.7)
        
        ax.set_title(f"{title} of Digits 1, 2, 4, 7")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("exercise_2/Q4_3_a_tsne_degrees_comparison.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # q4.2
    main_4_2()
    # q4.3
    main_4_3()