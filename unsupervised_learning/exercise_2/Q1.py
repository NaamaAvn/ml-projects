import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll
import warnings
warnings.filterwarnings("ignore")

def generate_swiss_roll(n_samples, noise, random_state):
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_state)
    # Normalize 't' to be between 0 and 1 for easier division
    t_normalized = (t - t.min()) / (t.max() - t.min())

    # in order to create discrete color sections as seen in the image
    # We'll use specific conditions on t_normalized to assign one of four colors
    segment_size = 1 / 4
    discrete_colors = ['red', 'green', 'blue', 'black']
    assigned_colors = []

    for val in t_normalized:
        if val < segment_size:
            assigned_colors.append(discrete_colors[0])
        elif val < 2 * segment_size:
            assigned_colors.append(discrete_colors[1])
        elif val < 3 * segment_size:
            assigned_colors.append(discrete_colors[2])
        else:
            assigned_colors.append(discrete_colors[3])
    return X, t, assigned_colors


def plot_swiss_roll(X, t, assigned_colors):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=assigned_colors, cmap=plt.cm.jet, s=20)
    ax.set_title("Swiss Roll Dataset with 4 Color Divisions")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    plt.savefig("Q1_a_swiss_roll.png")
    plt.show()



def main_1_a():
    # Generate the swiss roll dataset
    # n_samples: total number of points
    # noise: standard deviation of Gaussian noise added to the data
    # random_state: for reproducibility
    n_samples = 1500
    X, t, assigned_colors = generate_swiss_roll(n_samples, 0.1, 42)
    plot_swiss_roll(X, t, assigned_colors)


def generate_sinusoidal_curve():
    # Create 721 points from t=0 to t=8π with step π/90
    # t = 0: π/90 : 8π (MATLAB notation equivalent)
    step = np.pi / 90
    t = np.arange(0, 8*np.pi + step, step)
    
    # Define parametric equations
    X = t  # X(t) = t
    Y = t  # Y(t) = t  
    Z = np.sin(t)  # Z(t) = sin(t)
    return X, Y, Z, t

def plot_sinusoidal_curve(X, Y, Z, t):
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the 3D sinusoidal curve
    ax.plot(X, Y, Z, 'b-', linewidth=2)
    ax.scatter(X, Y, Z, c=t, cmap='viridis', s=10, alpha=0.6)
    
    # Highlight the origin point
    ax.scatter([0], [0], [0], color='red', s=100, marker='o', label='Origin (0,0,0)')
    
    # Force axis limits to show the full range including origin
    ax.set_xlim(0, 8*np.pi)
    ax.set_ylim(0, 8*np.pi)
    ax.set_zlim(-1.1, 1.1)
    
    # Remove auto margins
    ax.margins(0)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    ax.set_title("3D Sinusoidal Plot: X(t)=t, Y(t)=t, Z(t)=sin(t)")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis") 
    ax.set_zlabel("Z-axis")
    ax.legend()
    
    # Set viewing angle for better visualization
    ax.view_init(elev=20, azim=45)
    
    plt.savefig("Q1_b_sinusoidal_curve.png")
    plt.show()

def main_1_b():

    X, Y, Z, t = generate_sinusoidal_curve()
    plot_sinusoidal_curve(X, Y, Z, t)
    

if __name__ == "__main__":
    main_1_a()  
    main_1_b()
