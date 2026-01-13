import numpy as np
import pandas as pd
from itertools import combinations

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