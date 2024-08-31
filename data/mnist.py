import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_mnist_data(normalize=True):
    """
    Load and preprocess the MNIST dataset.

    Parameters:
    normalize (bool): Whether to normalize the pixel values to the range [0, 1].

    Returns:
    Tuple of arrays: (x_train_flat, x_test_flat, x_train_non_flat, x_test_non_flat, y_train, y_test)
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values to the range [0, 1] if specified
    if normalize:
        x_train = x_train / 255.0
        x_test = x_test / 255.0

    # Reshape the images into vectors of size 784 for methods like PCA, Isomap, and t-SNE
    x_train_flat = x_train.reshape(-1, 784)
    x_test_flat = x_test.reshape(-1, 784)

    # Keep the non-flattened images for CNNs and other deep learning models
    x_train_non_flat = x_train.reshape(-1, 28, 28, 1)
    x_test_non_flat = x_test.reshape(-1, 28, 28, 1)

    print("Flattened training matrix shape:", x_train_flat.shape)
    print("Flattened testing matrix shape:", x_test_flat.shape)
    print("Non-flattened training matrix shape:", x_train_non_flat.shape)
    print("Non-flattened testing matrix shape:", x_test_non_flat.shape)

    return x_train_flat, x_test_flat, x_train_non_flat, x_test_non_flat, y_train, y_test

def visualize_mnist_samples(x_train, y_train, num_samples=20):
    """
    Visualize a sample of images from the MNIST dataset.

    Parameters:
    x_train (ndarray): Training images (non-flattened).
    y_train (ndarray): Labels for the training images.
    num_samples (int): Number of samples to visualize.
    """
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(num_samples):
        ax = fig.add_subplot(4, 5, i + 1, xticks=[], yticks=[])
        ax.imshow(x_train[i].reshape(28, 28), cmap=plt.cm.binary)
        ax.text(10, -2.5, str(y_train[i]), color='red', fontsize=12)
    
    plt.show()

# # Example usage
# x_train_flat, x_test_flat, x_train_non_flat, x_test_non_flat, y_train, y_test = load_mnist_data(normalize=True)
# visualize_mnist_samples(x_train_non_flat, y_train, num_samples=20)
