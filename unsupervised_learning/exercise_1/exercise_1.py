import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import time
from itertools import combinations


def generate_clusters_a(S, C, W):
    """
    Creates M clusters of points based on Gaussian distribution.

    Parameters:
    - S (pd.Series): Series of size M with the number of points in each cluster.
    - C (pd.DataFrame): DataFrame of size M x D representing the centers of the clusters.
    - W (pd.Series): Series of size M with the width (standard deviation) of each cluster.

    Returns:
    - pd.DataFrame: DataFrame with D+1 columns, containing the coordinates of the points and the cluster number.
    """
    M = len(S)  # Number of clusters
    D = len(C.columns)  # Dimensions of the space

    # Prepare a list to store the points
    points = []

    # Create points for each cluster
    for i in range(M):
        num_points = S[i]  # Number of points in the i-th cluster
        center = C.iloc[i]  # The center of the i-th cluster
        width = W[i]  # The width (standard deviation) of the i-th cluster

        # Generate random points in [0, 1]^D space
        cluster_points = np.random.rand(num_points, D)

        # Move the points to the cluster center and add randomness based on the width
        cluster_points = cluster_points - 0.5  # Transform from [0,1] to [-1,1]
        cluster_points = cluster_points * width + center.values  # Scale by width and move to center

        # Add the cluster number to the points (for the last column)
        cluster_numbers = np.full(num_points, i + 1)

        # Combine the points and cluster numbers
        cluster_df = pd.DataFrame(cluster_points)
        cluster_df['Cluster'] = cluster_numbers

        points.append(cluster_df)

    # Concatenate all clusters into a single DataFrame
    final_df = pd.concat(points, ignore_index=True)

    return final_df


def plot_clusters(df, D=2, title="Cluster Visualization"):
    """
    Visualizes the clusters using a scatter plot. Each cluster is represented by a different color and shape.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the points and their corresponding cluster labels.
    - D (int): The number of dimensions to plot. Default is 2 for a 2D scatter plot.
    """
    # If D is 2, we can plot easily with scatter plot
    if D == 2:
        plt.figure(figsize=(8, 6))

        # Get unique cluster numbers
        clusters = df['Cluster'].unique()

        # Define different markers and colors for the clusters
        markers = ['o', 's', '^', 'D']
        colors = sns.color_palette("Set2", len(clusters))

        # Plot each cluster with different colors and markers
        for i, cluster in enumerate(clusters):
            cluster_data = df[df['Cluster'] == cluster]
            plt.scatter(cluster_data.iloc[:, 0],  # X-coordinate
                        cluster_data.iloc[:, 1],  # Y-coordinate
                        label=f"Cluster {cluster}",
                        color=colors[i],         # Color based on cluster
                        marker=markers[i % len(markers)],  # Shape based on index
                        edgecolors='k')  # Add edge color for visibility

        # Add labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.legend(title='Clusters')
        plt.grid(True)
        plt.show()

    else:
        print(f"Visualization is supported for 2D data. Dimensionality {D} is not supported for plotting")

def q_1_a(plot =False):
    S = pd.Series([100, 200, 100, 300])
    C = pd.DataFrame({
    'X': [1, 2, 4, 5],
    'Y': [1, 2, 2, 5]
})
    W = pd.Series([1, 1.3, 2, 3])

    df = generate_clusters_a(S, C, W)

    # Plot clusters
    if plot:
        plot_clusters(df)

    return df


def generate_clusters_b(S, C, D):
    """
    Creates M clusters in Gaussian distribution with the given parameters.

    Parameters:
    - S (pd.Series): Series of size M with the number of points in each cluster.
    - C (pd.DataFrame): DataFrame of size M x (D + D^2), where the first D columns represent the cluster centers
                        and the next D^2 columns represent the flattened covariance matrices.
    - D (int): The number of dimensions for the clusters.

    Returns:
    - pd.DataFrame: DataFrame with D+1 columns (coordinates and cluster label).
    """
    M = len(S)  # Number of clusters
    points = []  # List to hold all the points

    # Iterate over each cluster
    for i in range(M):
        # Extract the cluster center (μ) and covariance matrix (Σ)
        center = C.iloc[i, :D].values  # First D columns as the center
        cov_matrix_flat = C.iloc[i, D:].values  # The flattened covariance matrix
        cov_matrix = cov_matrix_flat.reshape(D, D)  # Reshape into D x D matrix

        # Generate S[i] random points in D-dimensional space
        num_points = S[i]

        # Generate random points from a standard normal distribution
        random_points = np.random.randn(num_points, D)

        # Apply the transformation p_new = Σ * p + μ
        transformed_points = random_points @ cov_matrix + center

        # Add the cluster number (i + 1) to the transformed points
        cluster_labels = np.full(num_points, i + 1)  # Cluster label for each point

        # Combine the points and labels into a DataFrame
        cluster_df = pd.DataFrame(transformed_points)
        cluster_df['Cluster'] = cluster_labels

        # Append the cluster's points to the list
        points.append(cluster_df)

    # Concatenate all the clusters into one DataFrame
    final_df = pd.concat(points, ignore_index=True)

    return final_df

def q_1_b(plot =False):
    # Number of clusters and points per cluster
    S = pd.Series([100, 200, 100, 300])

    # Cluster centers (D = 2) and covariance matrices (D^2 = 4 for D = 2)
    C = pd.DataFrame({
        'X1': [1, 1, 4, 5],
        'Y1': [1, 5, 2, 5],
        'X2': [0.3, 2, 2, 1],
        'Y2': [0, 0, 0.5, -0.9],
        'X3': [0, 0, 0.5, -0.9],
        'Y3': [0.3, 0.5, 0.55, 2],
    })

    # Create clusters in 2D space
    D = 2
    df = generate_clusters_b(S, C, D)

    # Plot clusters
    if plot:
        plot_clusters(df, D)

    return df



def generate_clusters_c(center, sigma, Ng, inner_radius, ring_width, Nr):
    """
    Generate a Gaussian cluster and a ring cluster around it.

    Parameters:
    - center: Tuple (x, y), center of both clusters
    - sigma: Standard deviation (not variance) of the Gaussian cluster
    - Ng: Number of points in the Gaussian cluster
    - inner_radius: Inner radius of the ring cluster
    - ring_width: Width of the ring
    - Nr: Number of points in the ring cluster

    Returns:
    - A (Ng + Nr, 3) numpy array: [x, y, label]
    """
    # Generate Gaussian cluster
    gaussian_points = np.random.normal(loc=center, scale=sigma, size=(Ng, 2))
    gaussian_labels = np.ones((Ng, 1))  # Label 1 for Gaussian cluster

    # Generate ring cluster
    r_min = inner_radius
    r_max = inner_radius + ring_width
    radii = np.sqrt(np.random.uniform(r_min ** 2, r_max ** 2, Nr))  # Uniform in area
    angles = np.random.uniform(0, 2 * np.pi, Nr)
    x_ring = center[0] + radii * np.cos(angles)
    y_ring = center[1] + radii * np.sin(angles)
    ring_points = np.column_stack((x_ring, y_ring))
    ring_labels = np.full((Nr, 1), 2)  # Label 2 for ring cluster

    # Combine clusters
    all_points = np.vstack((gaussian_points, ring_points))
    all_labels = np.vstack((gaussian_labels, ring_labels))

    return np.hstack((all_points, all_labels))

def q_1_c(plot =False):
    """
    Generate a Gaussian cluster and a ring cluster around it.
    """
    # Parameters
    center = (5, 5)
    sigma = 2
    Ng = 100
    inner_radius = 10
    ring_width = 2
    Nr = 200

    # Generate clusters
    data = generate_clusters_c(center, sigma, Ng, inner_radius, ring_width, Nr)

    # Create DataFrame for plotting
    df = pd.DataFrame(data, columns=['0', '1', 'Cluster'])

    # Plot clusters
    if plot:
        plot_clusters(df)

    return df


def inner_cluster_similarity(data: pd.DataFrame, labels: np.ndarray) -> float:
    """
    Computes the sum of distances between all pairs of points within each cluster.

    Parameters:
    - data: pd.DataFrame of shape (N, 2) with the coordinates
    - labels: np.ndarray of shape (N,) with cluster assignments

    Returns:
    - total_similarity: float, total intra-cluster distance
    """
    total_similarity = 0.0
    data_np = data.to_numpy()
    unique_labels = np.unique(labels)

    for label in unique_labels:
        cluster_points = data_np[labels == label]
        # Sum over all unique pairs in the cluster
        for i, j in combinations(range(len(cluster_points)), 2):
            dist = np.linalg.norm(cluster_points[i] - cluster_points[j])
            total_similarity += dist

    return total_similarity

def plot_similarities_scores(similarity_scores, title=""):
    """
    Print and lots the inner-cluster similarity scores against the number of clusters (k).

    Parameters:
    - similarity_scores: List of similarity scores for each k
    - k_values: List of k values corresponding to the similarity scores
    """
    k_values = list(range(1, len(similarity_scores) + 1))
    print(f"\nSimilarity scores:" + title)
    for k_val, sim in zip(k_values, similarity_scores):
        print(f"  k={k_val}: similarity = {sim:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(k_values, similarity_scores, marker='o')
    plt.title("Inner-cluster Similarity vs Number of Clusters (k)" + title)
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inner-cluster Similarity")
    plt.grid(True)
    plt.show()

def k_means(k=None, data=None, max_iters=100, tol=1e-4, init='random', eps=0.5, min_samples=5):
    """
    Performs K-means clustering on 2D data.

    Parameters:
    - k: int, number of clusters (ignored if init='dbscan')
    - data: pd.DataFrame of shape (N, 2)
    - max_iters: int, maximum number of iterations
    - tol: float, tolerance to stop updating centers
    - init: str, one of ['random', 'farthest', 'dbscan']
    - eps, min_samples: DBSCAN parameters (used if init='dbscan')

    Returns:
    - centers: np.ndarray of shape (k, 2), final cluster centers
    - labels: np.ndarray of shape (N,), cluster label for each point
    """
    np.random.seed(42)
    points = data.to_numpy()

    if init == 'random':
        if k is None:
            raise ValueError("k must be specified for 'random' init")
        initial_indices = np.random.choice(len(data), k, replace=False)
        centers = points[initial_indices]

    elif init == 'farthest':
        if k is None:
            raise ValueError("k must be specified for 'farthest' init")
        centers = [points[np.random.choice(len(points))]]
        while len(centers) < k:
            distances = np.min(np.linalg.norm(points[:, np.newaxis] - np.array(centers), axis=2), axis=1)
            next_center = points[np.argmax(distances)]
            centers.append(next_center)
        centers = np.array(centers)

    elif init == 'dbscan':
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        labels_db = db.labels_

        unique_labels = [label for label in set(labels_db) if label != -1]  # ignore noise
        k = len(unique_labels)
        if k == 0:
            raise ValueError("DBSCAN did not find any clusters")

        centers = np.array([
            points[labels_db == label].mean(axis=0)
            for label in unique_labels
        ])

    else:
        raise ValueError("init must be 'random', 'farthest', or 'dbscan'")

    # Run K-Means using these centers
    for _ in range(max_iters):
        distances = np.linalg.norm(points[:, np.newaxis] - centers, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centers = np.array([points[labels == i].mean(axis=0) for i in range(k)])

        if np.linalg.norm(new_centers - centers) < tol:
            break

        centers = new_centers

    return labels

def pam(k, data, max_iters=100):
    """
    Performs PAM (Partitioning Around Medoids) clustering.

    Parameters:
    - k: int, number of clusters
    - data: pd.DataFrame of shape (N, 2)
    - max_iters: int, maximum number of iterations

    Returns:
    - medoids: np.ndarray of shape (k, 2), final medoid points
    - labels: np.ndarray of shape (N,), cluster assignments
    """
    np.random.seed(42)
    points = data.to_numpy()
    n = len(points)

    # Step 1: Initialize medoids randomly
    initial_indices = np.random.choice(n, k, replace=False)
    medoids = points[initial_indices]

    for _ in range(max_iters):
        # Step 2: Assign each point to the nearest medoid
        distances = np.linalg.norm(points[:, np.newaxis] - medoids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Step 3: Update medoids
        new_medoids = []
        for i in range(k):
            cluster_points = points[labels == i]
            if len(cluster_points) == 0:
                # No points assigned to this cluster
                new_medoids.append(medoids[i])
                continue

            # Compute total cost for each point in cluster to be medoid
            costs = np.sum(np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2), axis=1)
            best_medoid = cluster_points[np.argmin(costs)]
            new_medoids.append(best_medoid)

        new_medoids = np.array(new_medoids)

        # Check for convergence
        if np.allclose(medoids, new_medoids):
            break

        medoids = new_medoids

    return medoids, labels

def pam_on_subset(k, subset_points, full_points):
    # Run PAM on subset
    n = len(subset_points)
    initial_indices = np.random.choice(n, k, replace=False)
    medoids = subset_points[initial_indices]

    for _ in range(100):
        distances = np.linalg.norm(subset_points[:, np.newaxis] - medoids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_medoids = []
        for i in range(k):
            cluster_points = subset_points[labels == i]
            if len(cluster_points) == 0:
                new_medoids.append(medoids[i])
                continue
            costs = np.sum(np.linalg.norm(cluster_points[:, np.newaxis] - cluster_points, axis=2), axis=1)
            best_medoid = cluster_points[np.argmin(costs)]
            new_medoids.append(best_medoid)

        new_medoids = np.array(new_medoids)
        if np.allclose(medoids, new_medoids):
            break
        medoids = new_medoids

    # Assign full data to medoids and compute cost
    distances_full = np.linalg.norm(full_points[:, np.newaxis] - medoids, axis=2)
    labels_full = np.argmin(distances_full, axis=1)
    total_cost = np.sum(np.min(distances_full, axis=1))

    return medoids, labels_full, total_cost

def clara(k, data, n_samples=5, sample_size=None):
    """
    CLARA (Clustering LARge Applications) implementation using PAM.

    Parameters:
    - k: int, number of clusters
    - data: pd.DataFrame of shape (N, 2)
    - n_samples: int, number of samples to draw
    - sample_size: int, size of each sample (default: min(40 + 2k, len(data)))

    Returns:
    - best_medoids: np.ndarray of shape (k, 2), best medoids found
    - best_labels: np.ndarray of shape (N,), final cluster assignments
    """
    np.random.seed(42)
    points = data.to_numpy()
    n = len(points)
    sample_size = sample_size or min(n, 40 + 2 * k)

    best_cost = float('inf')
    best_medoids = None
    best_labels = None

    for _ in range(n_samples):
        sample_indices = np.random.choice(n, sample_size, replace=False)
        sample_points = points[sample_indices]

        medoids, labels, cost = pam_on_subset(k, sample_points, points)

        if cost < best_cost:
            best_cost = cost
            best_medoids = medoids
            best_labels = labels

    return best_medoids, best_labels


if __name__ == "__main__":
    # Create datasets
    data_a = q_1_a()
    data_b = q_1_b()
    data_c = q_1_c()

    datasets = [data_a, data_b, data_c]

    # K means clustering for each dataset and each initialization method(random, farthest, dbscan)
    print("Running K-means clustering on datasets:")
    for i, dataset in enumerate(datasets, start=1):
        data = dataset.iloc[:, :2]  # Use the data without labels for clustering
        k_means_similarities_random = []  # To store similarity scores for each k - random init
        k_means_similarities_farthest = []  # To store similarity scores for each k - farthest init
        for k in range(1, 8):
            # Run K-means with random initialization
            print(f"Running K-means with k={k} on dataset {i} on random initialization")
            labels = k_means(k, data, init="random")
            # Calculate and store inner-cluster similarity
            similarity = inner_cluster_similarity(data, labels)
            k_means_similarities_random.append(similarity)
            plot_clusters(data.assign(Cluster=labels), title=f"Dataset {i} with {k} clusters - Random Init")
            # Run K-means with farthest initialization
            print(f"Running K-means with k={k} on dataset {i} on farthest initialization")
            labels = k_means(k, data, init="farthest")
            # Calculate and store inner-cluster similarity
            similarity = inner_cluster_similarity(data, labels)
            k_means_similarities_farthest.append(similarity)
            plot_clusters(data.assign(Cluster=labels), title=f"Dataset {i} with {k} clusters - Farthest Init")

        # Plot similarity scores for K-means
        plot_similarities_scores(k_means_similarities_random, title=f", dataset {i}, K-Means Random Initialization")
        plot_similarities_scores(k_means_similarities_farthest, title=f", dataset {i}, K-Means Farthest Initialization")

        # Run K-means with DBSCAN initialization
        print(f"\nRunning K-means with DBSCAN initialization on dataset {i}\n")
        labels = k_means(data=data, init="dbscan")
        plot_clusters(data.assign(Cluster=labels), title=f"Dataset {i} with DBSCAN initialization")


    # PAM clustering for each dataset
    start_pam = time.perf_counter()
    for i, dataset in enumerate(datasets, start=1):
        data = dataset.iloc[:, :2]  # Use the data without labels for clustering
        pam_similarities = []  # To store similarity scores for each k
        for k in range(1, 8):
            print(f"Running PAM with k={k} on dataset {i}")
            medoids, labels = pam(k, data)
            # Calculate and store inner-cluster similarity
            similarity = inner_cluster_similarity(data, labels)
            pam_similarities.append(similarity)
            # Plot clusters
            plot_clusters(data.assign(Cluster=labels), title=f"Dataset {i} with {k} clusters - PAM")
        # Plot similarity scores for PAM
        plot_similarities_scores(pam_similarities)
    end_pam = time.perf_counter()
    pam_time = end_pam - start_pam



    # CLARA clustering for each dataset
    start_clara = time.perf_counter()
    for i, dataset in enumerate(datasets, start=1):
        data = dataset.iloc[:, :2]  # Use the data without labels for clustering
        clara_similarities = []  # To store similarity scores for each k
        for k in range(1, 8):
            print(f"Running CLARA with k={k} on dataset {i}")
            medoids, labels = clara(k, data)
            # Calculate and store inner-cluster similarity
            similarity = inner_cluster_similarity(data, labels)
            clara_similarities.append(similarity)
            # Plot clusters
            plot_clusters(data.assign(Cluster=labels), title=f"Dataset {i} with {k} clusters - CLARA")
        # Plot similarity scores for CLARA
        plot_similarities_scores(clara_similarities)
    end_clara = time.perf_counter()
    clara_time = end_clara - start_clara


    # Compare runtimes of PAM and CLARA assuming plotting time is identical for both
    print(f"PAM runtime:   {pam_time:.4f} seconds")
    print(f"CLARA runtime: {clara_time:.4f} seconds")