import numpy as np
from numba import jit
import gc
import psutil
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Mean
from memory_profiler import memory_usage
import faiss
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import shortest_path

# Function to get current memory usage
def memory_usage_psutil():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    return mem

# Import the autoencoder creation functions from the respective .py files
from models.dense_autoencoder import get_dense_autoencoder
from models.cnn_autoencoder import get_cnn_autoencoder
from models.text_autoencoder import get_text_autoencoder

class GCAETrainer(Model):
    """
    A trainer class for the GCAE that handles model training and evaluation.
    """
    def __init__(self, dataset, model_type="dense", input_dim=3, embedding_dim=2, n_neighbors=10, alpha=1, seed=1, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.dataset = dataset
        self.model_type = model_type  # Attribute to select architecture
        self.n_neighbors = n_neighbors
        self.alpha = alpha
        self.seed = seed
        self.gcae = self.get_autoencoder()  # Autoencoder based on model_type
        if self.dataset == "mnist":
            self.encoder = self.gcae.get_layer(f'{self.dataset}_cnn_encoder')
        else:
            self.encoder = self.gcae.get_layer(f'{self.dataset}_encoder')

        # FAISS index and training data
        self.faiss_index = None
        self.neigh_graph = None
        self.train_geo_dist = None
        self.train_geo_dist_max = None
        self.test_geo_dist = None
        
        # Define the loss metrics to track and log
        self.total_loss_metric = Mean(name='total_loss')
        self.reconstruction_loss_metric = Mean(name='reconstruction_loss')
        self.geodesic_loss_metric = Mean(name='geodesic_loss')

    def get_autoencoder(self):
        """
        Returns the selected autoencoder based on the model type.
        """
        if self.model_type == "dense":
            return get_dense_autoencoder(self.input_dim, self.embedding_dim, self.dataset, seed=self.seed)
        elif self.model_type == "cnn":
            input_shape = (28, 28, 1)  # Example input shape for MNIST-like data
            return get_cnn_autoencoder(input_shape=input_shape, encoded_dim=self.embedding_dim, dataset=self.dataset, seed=self.seed)
        elif self.model_type == "text":
            return get_text_autoencoder(self.input_dim, self.embedding_dim, self.dataset, seed=self.seed)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    @property
    def metrics(self):
        """
        Returns a list of metrics used in training and evaluation.
        """
        return [self.total_loss_metric, self.reconstruction_loss_metric, self.geodesic_loss_metric]

    @staticmethod
    @jit(nopython=True)
    def construct_graph(indices, distances, n_neighbors, n_samples):
        graph = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(1, n_neighbors):  # start from 1 to skip self-loop
                graph[i, indices[i, j]] = np.sqrt(distances[i, j])
                graph[indices[i, j], i] = np.sqrt(distances[i, j])
        return graph

    @staticmethod
    @jit(nopython=True)
    def construct_expanded_graph(indices, distances, neigh_graph, n_neighbors, n_train, n_samples):
        n_total = n_train + n_samples
        graph = np.zeros((n_total, n_total))
        graph[:n_train, :n_train] = neigh_graph
        for i in range(n_samples):
            for j in range(1, n_neighbors):  # start from 1 to skip self-loop
                graph[n_train+i, indices[i, j]] = np.sqrt(distances[i, j])
                graph[indices[i, j], n_train+i] = np.sqrt(distances[i, j])
        return graph

    def compute_geodesic_distances(self, data, training=False):
    
        if training:
            # Use FAISS to find approximate nearest neighbors
            d = data.shape[1]
            faiss_index = faiss.IndexFlatL2(d)
            faiss_index.add(data)
            distances, indices = faiss_index.search(data, self.n_neighbors)
            self.faiss_index = faiss_index
    
            # Construct sparse neighborhood graph
            n_samples = data.shape[0]
            print(f"Memory usage before constructing graph: {memory_usage_psutil()} MB")
            graph = self.construct_graph(indices, distances, self.n_neighbors, n_samples)
            self.neigh_graph = graph
            graph = csr_matrix(graph)
            print(f"Memory usage after constructing graph: {memory_usage_psutil()} MB")
    
            # Compute geodesic distances using the shortest path algorithm
            print(f"Memory usage before shortest path: {memory_usage_psutil()} MB")
            geodesic_distances = shortest_path(csgraph=graph, method='D', directed=False)
            print(f"Memory usage after shortest path: {memory_usage_psutil()} MB")
            geodesic_distances = np.float32(geodesic_distances)
            self.train_geo_dist = tf.convert_to_tensor(geodesic_distances)
            self.train_geo_dist_max = np.max(geodesic_distances)

        else:
            # Use trained FAISS index to find approximate nearest neighbors
            n_train = self.faiss_index.ntotal
            n_samples = data.shape[0]
            print(f"Memory usage before searching: {memory_usage_psutil()} MB")
            distances, indices = self.faiss_index.search(data, self.n_neighbors)
    
            # Construct sparse neighborhood graph
            print(f"Memory usage before constructing expanded graph: {memory_usage_psutil()} MB")
            graph = self.construct_expanded_graph(indices, distances, self.neigh_graph, self.n_neighbors, n_train, n_samples)
            graph = csr_matrix(graph)
            print(f"Memory usage after constructing expanded graph: {memory_usage_psutil()} MB")
    
            # Compute geodesic distances using the shortest path algorithm for all data
            print(f"Memory usage before shortest path: {memory_usage_psutil()} MB")
            geodesic_distances = shortest_path(csgraph=graph, method='D', directed=False)
            print(f"Memory usage after shortest path: {memory_usage_psutil()} MB")
    
            # Extract the geodesic distances matrix for only new data
            geodesic_distances = geodesic_distances[n_train:, n_train:]
            geodesic_distances = np.float32(geodesic_distances)
            self.test_geo_dist = tf.convert_to_tensor(geodesic_distances)
        
        gc.collect()
        
        return geodesic_distances
    
    def compute_geodesic_loss(self, z, geodesic_distances, batch_indices):
        """
        Computes geodesic distance loss.
        """
        # Calculate the pairwise Euclidean distances in the latent space
        z_distances = tf.norm(z[:, tf.newaxis] - z, axis=2)
        # Calculate the geodesic distance loss
        geodesic_distances_batch = tf.gather(tf.gather(geodesic_distances, batch_indices, axis=0), batch_indices, axis=1)
        distance_loss = tf.reduce_mean((z_distances - geodesic_distances_batch) ** 2) / self.train_geo_dist_max
        return distance_loss
        
    def _get_losses(self, x, batch_indices, training=False):
        """
        Computes model losses from inputs.
        """
        reconstructions = self.gcae(x, training=training)
        latent = self.encoder(x)
        # Compute the reconstruction loss
        reconstruction_loss = (
            tf.reduce_mean((x - reconstructions) ** 2)
        )
        # Compute the geodesic loss
        if training:
            geodesic_loss = self.compute_geodesic_loss(z=latent, geodesic_distances=self.train_geo_dist, 
                                                       batch_indices=batch_indices)
        else:    
            geodesic_loss = self.compute_geodesic_loss(z=latent, geodesic_distances=self.test_geo_dist, 
                                                       batch_indices=batch_indices)
        # Cmpute the total loss
        total_loss = reconstruction_loss + self.alpha * geodesic_loss
        return total_loss, reconstruction_loss, geodesic_loss

    def train_step(self, data):
        """
        Performs one training step using a single batch of data.
        """
        x, batch_indices = data
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, geodesic_loss = self._get_losses(x, batch_indices, training=True)
        grads = tape.gradient(total_loss, self.gcae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.gcae.trainable_variables))

        # Update loss metrics
        self.total_loss_metric.update_state(total_loss)
        self.reconstruction_loss_metric.update_state(reconstruction_loss)
        self.geodesic_loss_metric.update_state(geodesic_loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        """
        Evaluates the model using a single batch of data.
        """
        x, batch_indices = data
        total_loss, reconstruction_loss, geodesic_loss = self._get_losses(x, batch_indices, training=False)

        # Update loss metrics
        self.total_loss_metric.update_state(total_loss)
        self.reconstruction_loss_metric.update_state(reconstruction_loss)
        self.geodesic_loss_metric.update_state(geodesic_loss)
        return {m.name: m.result() for m in self.metrics}