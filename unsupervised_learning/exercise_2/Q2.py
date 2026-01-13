import numpy as np
import matplotlib.pyplot as plt
from Q1 import generate_swiss_roll, generate_sinusoidal_curve
from exercise_1.Q1.Q1 import q_1_b
from sklearn.datasets import load_digits
from sklearn.cluster import MeanShift
from sklearn.metrics import silhouette_score

def pca_a_N_bigger_than_D(X, K=2):
    """
    PCA using SVD while N >> D
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Perform SVD
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    # Vt contains the principal directions (rows)
    V = Vt.T  # shape (D, D)
    singular_values = S  # shape (D,)

    # Project data onto top K components
    Z = X_centered @ V[:, :K]  # shape: (N, K)

    return Z


def pca_b_D_bigger_than_N(X, K=2):
    """
    PCA using SVD while D >> N
    """
    # Center the data
    X_centered = X - np.mean(X, axis=0)

    # Compute the Gram matrix (N x N)
    gram_matrix = X_centered @ X_centered.T  # shape: (N, N)

    # Perform SVD on K
    U_small, S_small, _ = np.linalg.svd(gram_matrix)

    # Compute principal components in original D-dimensional space
    #    (Use eigenvectors of X^T X)
    # Each PC = X^T @ u_i / sqrt(s_i)
    components = (X_centered.T @ U_small) / np.sqrt(S_small + 1e-10)  # shape: (D, N)

    # Normalize (optional)
    N = X.shape[0]
    components = components[:, :N]  # keep top N components
    components = components / np.linalg.norm(components, axis=0)

    # Project data into K-dimensional PCA space
    Z = X_centered @ components[:, :K]  # shape: (N, K)

    return Z

def pca_swiss_roll():
    """
    PCA on swiss roll dataset
    """
    # Swiss roll dataset
    X_swiss_roll, t, assigned_colors = generate_swiss_roll(1500, 0.1, 42)
    z_swiss_roll_a = pca_a_N_bigger_than_D(X_swiss_roll)
    z_swiss_roll_b = pca_b_D_bigger_than_N(X_swiss_roll)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(z_swiss_roll_a[:, 0], z_swiss_roll_a[:, 1], c=assigned_colors)
    plt.title("Swiss Roll Dataset (PCA Method A, N >> D)")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(z_swiss_roll_b[:, 0], z_swiss_roll_b[:, 1], c=assigned_colors)
    plt.title("Swiss Roll Dataset (PCA Method B, D >> N)")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    
    plt.tight_layout()
    plt.savefig("exercise_2/Q2_3_swiss_roll.png", dpi=300, bbox_inches='tight')
    # plt.show()

def pca_sinusoidal_curve():
    """
    PCA on sinusoidal curve dataset
    """
    # Sinusoidal curve dataset
    X_coords, Y_coords, Z_coords, t = generate_sinusoidal_curve()
    # Combine coordinates into a single 2D array for PCA
    X_sinusoidal = np.column_stack([X_coords, Y_coords, Z_coords])  # Shape: (N, 3)
    z_sinusoidal_a = pca_a_N_bigger_than_D(X_sinusoidal)
    z_sinusoidal_b = pca_b_D_bigger_than_N(X_sinusoidal)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    scatter3 = plt.scatter(z_sinusoidal_a[:, 0], z_sinusoidal_a[:, 1], c=t, cmap='plasma')
    plt.title("Sinusoidal Curve Dataset (PCA Method A, N >> D)")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    
    plt.subplot(1, 2, 2)
    scatter4 = plt.scatter(z_sinusoidal_b[:, 0], z_sinusoidal_b[:, 1], c=t, cmap='plasma')
    plt.title("Sinusoidal Curve Dataset (PCA Method B, D >> N)")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    
    plt.tight_layout()
    plt.savefig("exercise_2/Q2_3_sinusoidal.png", dpi=300, bbox_inches='tight')
    # plt.show()

def pca_5_5():
    """
    PCA on 5,5 dataset
    """
    # 5,5 dataset
    df = q_1_b()
    X = df[df.Cluster == 4].values[:, :2]
    z_5_5_a = pca_a_N_bigger_than_D(X, K=1)
    z_5_5_b = pca_b_D_bigger_than_N(X, K=1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # Create 1D scatter plot with some jitter for better visualization
    y_jitter_a = np.random.normal(0, 0.01, len(z_5_5_a))
    scatter5 = plt.scatter(z_5_5_a[:, 0], y_jitter_a)
    plt.title("5,5 Dataset (PCA Method A, N >> D)")
    plt.xlabel("First Principal Component")
    plt.ylabel("Random Jitter (for visualization)")
    plt.ylim(-0.5, 0.5)
    
    plt.subplot(1, 2, 2)
    # Create 1D scatter plot with some jitter for better visualization
    y_jitter_b = np.random.normal(0, 0.01, len(z_5_5_b))
    scatter6 = plt.scatter(z_5_5_b[:, 0], y_jitter_b)
    plt.title("5,5 Dataset (PCA Method B, D >> N)")
    plt.xlabel("First Principal Component")
    plt.ylabel("Random Jitter (for visualization)")
    plt.ylim(-0.5, 0.5)
    
    plt.tight_layout()  
    plt.savefig("exercise_2/Q2_3_5_5.png", dpi=300, bbox_inches='tight')
    # plt.show()

def pca_D_0_3_8():
    """
    PCA on D_0_3_8 Mnist dataset
    """
    #  D_0_3_8 Mnist dataset
    digits = load_digits()
    # filter out the digits that are not 0, 3, 8
    mask = np.isin(digits.target, [0, 3, 8])
    X = digits.data[mask]
    y = digits.target[mask]
    z_0_3_8_a = pca_a_N_bigger_than_D(X, K=2)
    z_0_3_8_b = pca_b_D_bigger_than_N(X, K=2)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    scatter7 = plt.scatter(z_0_3_8_a[:, 0], z_0_3_8_a[:, 1], c=y, cmap='viridis')
    cbar1 = plt.colorbar(scatter7, label='Digit Label')
    cbar1.set_ticks([0, 3, 8])
    cbar1.set_ticklabels(['0', '3', '8'])
    plt.title("Mnist Dataset (PCA Method A, N >> D)")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")

    plt.subplot(1, 2, 2)
    scatter8 = plt.scatter(z_0_3_8_b[:, 0], z_0_3_8_b[:, 1], c=y, cmap='viridis')
    cbar2 = plt.colorbar(scatter8, label='Digit Label')
    cbar2.set_ticks([0, 3, 8])
    cbar2.set_ticklabels(['0', '3', '8'])
    plt.title("Mnist Dataset (PCA Method B, D >> N)")
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")

    plt.tight_layout()
    plt.savefig("exercise_2/Q2_3_D_0_3_8.png", dpi=300, bbox_inches='tight')
    # plt.show()

    return X, y, z_0_3_8_a, z_0_3_8_b

def mean_shift_analysis_D_0_3_8(X, y, z_0_3_8_a, z_0_3_8_b):
    """
    Mean Shift analysis on D_0_3_8 dataset
    """
    # Mean Shift analysis on D_0_3_8 dataset
    print("\n=== Mean Shift Analysis on D_0_3_8 Dataset ===")
    
    # Use both original data and PCA result
    X_original = X  # Original high-dimensional data (64 features)
    X_pca_2d = z_0_3_8_a  # 2D PCA projection (method A because N >> D)
    
    print(f"Original data shape: {X_original.shape}")
    print(f"PCA data shape: {X_pca_2d.shape}")
    
    # Define different bandwidth values for different dimensionality
    r_values_pca = [1, 5, 10, 13, 15]  # For 2D PCA data
    r_values_original = [15, 30, 32, 33, 35]  # For 64D original data (much larger)
    
    # Perform Mean Shift analysis on original data
    print("\n--- Original High-Dimensional Data (64D) ---")
    print(f"Using bandwidth values: {r_values_original}")
    similarities_original, n_clusters_original = mean_shift_analysis(X_original, r_values_original)
    
    # Perform Mean Shift analysis on PCA data
    print("\n--- PCA-Reduced Data (2D) ---")
    print(f"Using bandwidth values: {r_values_pca}")
    similarities_pca, n_clusters_pca = mean_shift_analysis(X_pca_2d, r_values_pca)
    
    # Plot comparison between original and PCA data
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(r_values_original, similarities_original, 'ro-', linewidth=2, markersize=8, label='Original Data (64D)')
    plt.plot(r_values_pca, similarities_pca, 'bo-', linewidth=2, markersize=8, label='PCA Data (2D)')
    plt.xlabel('Bandwidth (r)')
    plt.ylabel('Cluster Inner Similarity (Silhouette Score)')
    plt.title('Mean Shift: Original vs PCA Data Similarity')
    plt.legend()
    plt.grid(True, alpha=0.3)

    
    plt.tight_layout()
    plt.savefig("exercise_2/Q2_mean_shift_original_vs_pca.png", dpi=300, bbox_inches='tight')
    # plt.show()
    
    # Find best bandwidth for each approach
    best_r_original = r_values_original[np.argmax(similarities_original)]
    best_r_pca = r_values_pca[np.argmax(similarities_pca)]
    best_similarity_original = max(similarities_original)
    best_similarity_pca = max(similarities_pca)
    print(f"\nBest bandwidth for Original Data: r = {best_r_original} (similarity = {best_similarity_original:.3f})")
    print(f"Best bandwidth for PCA Data: r = {best_r_pca} (similarity = {best_similarity_pca:.3f})")
    
    if best_similarity_original > best_similarity_pca:
        print("→ Original high-dimensional data generally performs better for clustering")
    else:
        print("→ PCA-reduced data generally performs better for clustering")

def calculate_cluster_inner_similarity(X, labels):
    """
    Calculate the average inner-cluster similarity (silhouette score)
    """
    if len(np.unique(labels)) < 2:
        return 0  # Need at least 2 clusters for silhouette score
    return silhouette_score(X, labels)


def mean_shift_analysis(X_pca, r_values):
    """
    Apply Mean Shift clustering with different bandwidth values and calculate similarities
    """
    similarities = []
    n_clusters_list = []
    
    for r in r_values:
        # Apply Mean Shift with bandwidth r
        mean_shift = MeanShift(bandwidth=r)
        cluster_labels = mean_shift.fit_predict(X_pca)
        
        # Calculate inner similarity
        similarity = calculate_cluster_inner_similarity(X_pca, cluster_labels)
        similarities.append(similarity)
        n_clusters_list.append(len(np.unique(cluster_labels)))
        
        print(f"r = {r:.2f}: {len(np.unique(cluster_labels))} clusters, similarity = {similarity:.3f}")
    
    return similarities, n_clusters_list


def main_2_3():
    # Perform PCA on swiss roll dataset
    pca_swiss_roll()

    # Perform PCA on sinusoidal curve dataset
    pca_sinusoidal_curve()

    # Perform PCA on last exercise's 5,5 dataset
    pca_5_5()

    # Perform PCA on D_0_3_8 dataset
    X, y, z_0_3_8_a, z_0_3_8_b = pca_D_0_3_8()

    # Perform Mean Shift analysis on D_0_3_8 dataset
    mean_shift_analysis_D_0_3_8(X, y, z_0_3_8_a, z_0_3_8_b)



if __name__ == "__main__":
    
    # Run all required question 2.3 parts
    main_2_3()
