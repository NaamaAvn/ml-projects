import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
        # plt.savefig(f"/Users/naamaavni/workspace/naama/unsupervised_learning/exercise_1/Q4/{title}")

    else:
        print(f"Visualization is supported for 2D data. Dimensionality {D} is not supported for plotting")

def q_1_a(plot =False):
    """
    Creates M clusters of points on D dimention based on Gaussian distribution.
    """
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

if __name__ == "__main__":
    # Create datasets
    data_a = q_1_a(plot=True)
    data_b = q_1_b(plot=True)
    data_c = q_1_c(plot=True)