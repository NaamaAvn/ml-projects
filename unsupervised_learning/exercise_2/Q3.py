import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from Q2 import generate_swiss_roll, generate_sinusoidal_curve
from exercise_1.Q1.Q1 import q_1_b
from sklearn.cluster import MeanShift
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import rbf_kernel

def lle(X, K=10, n_components=2, reg=1e-3):
    """
    Perform Locally Linear Embedding (LLE).

    Parameters:
    - X: Input data (N x D)
    - K: Number of neighbors to use
    - n_components: Dimension of the reduced embedding
    - reg: Regularization for numerical stability

    Returns:
    - Y: Embedded coordinates (N x n_components)
    """

    N, D = X.shape

    # Step 1: Find nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=K + 1).fit(X)
    distances, indices = nbrs.kneighbors(X)
    neighbors = indices[:, 1:]  # exclude the point itself

    # Step 2: Compute reconstruction weights
    W = np.zeros((N, N))

    for i in range(N):
        Z = X[neighbors[i]] - X[i]  # shape: (K, D)
        C = Z @ Z.T  # local covariance
        C += reg * np.trace(C) * np.eye(K)  # regularization
        w = np.linalg.solve(C, np.ones(K))
        w /= np.sum(w)
        W[i, neighbors[i]] = w

    # Step 3: Compute embedding from eigenvectors of (I - W)^T(I - W)
    M = np.eye(N) - W
    M = M.T @ M

    # Step 4: Compute bottom (non-zero) eigenvectors
    # We skip the smallest eigenvector (which is all ones)
    eigvals, eigvecs = eigsh(M, k=n_components + 1, sigma=0.0, which='LM')
    Y = eigvecs[:, 1:n_components + 1]

    return Y

def main_3_2():
    # Swiss roll dataset
    n_samples = 1500
    X, t, assigned_colors = generate_swiss_roll(n_samples, 0.1, 42)

    Y = lle(X, K=10, n_components=2, reg=1e-3)

    plt.figure()  # Create new figure for Swiss roll plot
    plt.scatter(Y[:, 0], Y[:, 1], c=assigned_colors, s=5)
    plt.title("LLE Embedding")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig("exercise_2/Q3_2_lle_swiss_roll.png", dpi=300, bbox_inches='tight')

    # Sinusoidal curve dataset
    X_coords, Y_coords, Z_coords, t = generate_sinusoidal_curve()
    # Combine coordinates into a single 2D array for PCA
    X_sinusoidal = np.column_stack([X_coords, Y_coords, Z_coords])  # Shape: (N, 3)

    Y = lle(X_sinusoidal, K=10, n_components=2, reg=1e-3)

    plt.figure()  # Create new figure for sinusoidal plot
    plt.scatter(Y[:, 0], Y[:, 1], c=t, s=5)  # Use t from sinusoidal curve for coloring
    plt.title("LLE Embedding")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig("exercise_2/Q3_2_lle_sinusoidal.png", dpi=300, bbox_inches='tight')

    # 5,5 dataset
    df = q_1_b()
    X = df[df.Cluster == 4].values[:, :2]

    Y = lle(X, K=10, n_components=2, reg=1e-3)

    plt.figure()  # Create new figure for 5,5 dataset plot
    plt.scatter(Y[:, 0], Y[:, 1], s=5)
    plt.ylim(-0.5, 0.5)
    plt.title("LLE Embedding")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.savefig("exercise_2/Q3_2_lle_5_5.png", dpi=300, bbox_inches='tight')


def main_3_3():
    # D_1_2_4_7 dataset
    digits = load_digits()
    # filter out the digits that are not 1, 2, 4, 7
    mask = np.isin(digits.target, [1, 2, 4, 7])
    X = digits.data[mask]
    y = digits.target[mask]

    # Create a figure with subplots for all K values
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle("LLE Embedding - Digits 1, 2, 4, 7 with Different K Values", fontsize=16)
    
    for idx, K in enumerate([10, 15, 20, 25, 30]):
        Y = lle(X, K=K, n_components=2, reg=1e-3)

        ax = axes[idx]
        # Create separate scatter plots for each digit to enable legend
        colors = ['red', 'blue', 'green', 'orange']
        digits_to_plot = [1, 2, 4, 7]
        
        for i, digit in enumerate(digits_to_plot):
            mask_digit = (y == digit)
            ax.scatter(Y[mask_digit, 0], Y[mask_digit, 1], 
                      c=colors[i], s=5, label=f'Digit {digit}')
        
        ax.set_title(f"K={K}")
        ax.set_xlabel("Component 1")   
        ax.set_ylabel("Component 2")
        
        # Only add legend to the first subplot to avoid clutter
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()
    plt.savefig("exercise_2/Q3_3_lle_D_1_2_4_7_all_K.png", dpi=300, bbox_inches='tight')

    # Run Mean Shift on D_1_2_4_7 dataset with best K value (20)
    best_K = 20
    best_Y = lle(X, K=best_K, n_components=2, reg=1e-3)
    
    for r in [0.01, 0.02, 0.03, 0.04, 0.05]:
        mean_shift = MeanShift(bandwidth=r)
        y_pred = mean_shift.fit_predict(best_Y)
        print(f"Bandwidth(r): {r}")
        print(f"Number of clusters: {len(np.unique(y_pred))}")
        print(f"Similarity(Silhouette Score): {round(silhouette_score(best_Y, y_pred), 3)}")

        # calculate Ncut
        # Step 1: Build similarity graph using RBF kernel
        similarity_matrix = rbf_kernel(best_Y, gamma=1.0 / (2 * (r ** 2)))  # same r used in MeanShift as bandwidth

        # Step 2: Calculate Ncut score
        ncut_score = 0
        labels = np.unique(y_pred)

        for label in labels:
            idx_in = np.where(y_pred == label)[0]
            idx_out = np.where(y_pred != label)[0]

            cut = np.sum(similarity_matrix[np.ix_(idx_in, idx_out)])
            vol = np.sum(similarity_matrix[idx_in, :])

            if vol > 0:
                ncut_score += cut / vol

        print(f"Ncut Score: {round(ncut_score, 4)}")
        print("------")


    # create a K * R * Ncut plot
    ncut_scores = []
    r_values = []
    k_values = []

    for idx, K in enumerate([10, 15, 20, 25, 30]):
        Y = lle(X, K=K, n_components=2, reg=1e-3)

        for r in [0.01, 0.02, 0.03, 0.04, 0.05]:
            mean_shift = MeanShift(bandwidth=r)
            y_pred = mean_shift.fit_predict(Y)
            
            # Check if we have more than 1 cluster
            n_clusters = len(np.unique(y_pred))
            if n_clusters <= 1:
                print(f"K: {K}, r: {r}, Number of clusters: {n_clusters} - Skipping (insufficient clusters)")
                print("------")
                continue
            
            similarity_matrix = rbf_kernel(Y, gamma=1.0 / (2 * (r ** 2))) 

            ncut_score = 0
            labels = np.unique(y_pred)

            for label in labels:
                idx_in = np.where(y_pred == label)[0]
                idx_out = np.where(y_pred != label)[0]

                cut = np.sum(similarity_matrix[np.ix_(idx_in, idx_out)])
                vol = np.sum(similarity_matrix[idx_in, :])

                if vol > 0:
                    ncut_score += cut / vol

            # save values
            ncut_scores.append(ncut_score)
            r_values.append(r)
            k_values.append(K)

            print(f"K: {K}, r: {r}, Number of clusters: {n_clusters}, Ncut Score: {round(ncut_score, 4)}")
            print("------")

    # create a 3D scatter plot and highlight the best K and r and the best Ncut Score
    if len(ncut_scores) == 0:
        print("No valid clustering results found. All combinations produced single clusters.")
        return
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(k_values, r_values, ncut_scores)
    ax.scatter(k_values[np.argmin(ncut_scores)], r_values[np.argmin(ncut_scores)], ncut_scores[np.argmin(ncut_scores)], color='red', s=100, label='Best K and r')
    ax.set_xlabel("K")
    ax.set_ylabel("r")
    ax.set_zlabel("Ncut Score")
    ax.legend()
    plt.savefig("exercise_2/Q3_3_lle_D_1_2_4_7_K_R_Ncut.png", dpi=300, bbox_inches='tight')
    plt.show()

    # print the best K and r and the best Ncut Score
    print(f"Best K: {k_values[np.argmin(ncut_scores)]}, Best r: {r_values[np.argmin(ncut_scores)]}, Best Ncut Score: {ncut_scores[np.argmin(ncut_scores)]}")

if __name__ == "__main__":
    # q3.2
    main_3_2()
    # q3.3
    main_3_3()