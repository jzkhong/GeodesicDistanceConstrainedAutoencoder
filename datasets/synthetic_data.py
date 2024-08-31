import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import ticker

# Set seed for reproducibility
seed = 42
np.random.seed(seed)

def load_synthetic_data(dataset_type='swiss_roll', n_samples=4000, noise=0.1, test_size=0.2, seed=42):
    """
    Generate and load synthetic data (Swiss Roll or S-Curve) and split it into training and test sets.

    Parameters:
    dataset_type (str): Type of dataset to generate ('swiss_roll' or 's_curve').
    n_samples (int): Number of samples to generate.
    noise (float): Standard deviation of Gaussian noise added to the data.
    test_size (float): Proportion of the dataset to include in the test split.
    seed (int): Random seed for reproducibility.

    Returns:
    Tuple of arrays: (x_train_scaled, x_test_scaled, y_train, y_test)
    """
    if dataset_type == 'swiss_roll':
        X, y = datasets.make_swiss_roll(n_samples, noise=noise, random_state=seed)
    elif dataset_type == 's_curve':
        X, y = datasets.make_s_curve(n_samples, noise=noise, random_state=seed)
    else:
        raise ValueError("Invalid dataset_type. Choose either 'swiss_roll' or 's_curve'.")

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # Standardize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Shape of {dataset_type.replace('_', ' ').capitalize()} dataset:", X_train.shape)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def plot_3d(points, points_color, title, cmap=plt.cm.viridis):
    """
    Generate a 3D scatter plot of the data.

    Parameters:
    points (ndarray): Array of data points with shape (n_samples, 3).
    points_color (ndarray): Array of colors or labels for each data point.
    title (str): Title of the plot.
    cmap (Colormap): Colormap used for coloring the points.
    """
    x, y, z = points.T
    fig = plt.figure(figsize=(6, 6)) 
    ax = fig.add_subplot(111, projection='3d') 
    col = ax.scatter(x, y, z, c=points_color, cmap=cmap, alpha=0.6)
    ax.view_init(azim=-80, elev=12)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

# # Example usage for Swiss Roll dataset
# x_swiss_scaled, x_test_swiss_scaled, y_swiss, y_test_swiss = load_synthetic_data('swiss_roll', n_samples=4000, noise=0.1, test_size=0.2, seed=seed)
# plot_3d(x_swiss_scaled, y_swiss, 'Swiss Roll Dataset', cmap=plt.cm.jet)

# # Example usage for S-Curve dataset
# x_s_scaled, x_test_s_scaled, y_s, y_test_s = load_synthetic_data('s_curve', n_samples=4000, noise=0.1, test_size=0.2, seed=seed)
# plot_3d(x_s_scaled, y_s, 'S-Curve Dataset', cmap=plt.cm.viridis)
