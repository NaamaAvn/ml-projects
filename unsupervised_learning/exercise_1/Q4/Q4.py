import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from Q1.Q1 import q_1_a, q_1_b, q_1_c, plot_clusters
from Q2.Q2 import inner_cluster_similarity
from Q3.Q3 import plot_similarities_scores


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