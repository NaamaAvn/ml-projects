import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from Q1.Q1 import q_1_a, q_1_b, q_1_c, plot_clusters
from Q2.Q2 import inner_cluster_similarity


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
    # plt.savefig(f"/Users/naamaavni/workspace/naama/unsupervised_learning/exercise_1/Q4/Inner-cluster Similarity vs Number of Clusters (k) {title}")


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

        # Plot similarity scores for PAM
        plot_similarities_scores(k_means_similarities_random, title=f", dataset {i}, K-Means Random Initialization")
        plot_similarities_scores(k_means_similarities_farthest, title=f", dataset {i}, K-Means Farthest Initialization")

        # Run K-means with DBSCAN initialization
        print(f"\nRunning K-means with DBSCAN initialization on dataset {i}\n")
        labels = k_means(data=data, init="dbscan")
        plot_clusters(data.assign(Cluster=labels), title=f"Dataset {i} with DBSCAN initialization")
