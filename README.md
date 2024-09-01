# Geodesic Distance Constrained Autoencoder (GCAE)

This repository contains the code implementation for the Geodesic Distance Constrained Autoencoder (GCAE) developed as part of a MSc research project. The GCAE is designed to incorporate geodesic distances within the latent space of autoencoders to preserve the intrinsic geometry of the data.

## Table of Contents
- [Geodesic Distance Constrained Autoencoder (GCAE)](#geodesic-distance-constrained-autoencoder-gcae)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Datasets](#datasets)
  - [Models](#models)
  - [Example Usage](#example-usage)
  - [Dependencies](#dependencies)
  - [Code License](#code-license)

## Introduction

The GCAE framework is implemented to explore the impact of incorporating geodesic distance constraints in autoencoder architectures. The models included in this repository are specifically tailored for various datasets, including synthetic data, MNIST, and the 12-Newsgroups text data.

## Datasets

The following datasets are used in this study, with corresponding data loading and preprocessing scripts available in the `data` folder:

- **Synthetic Data**: Includes 3D Swiss Roll and S-Curve datasets.
- **MNIST**: 70,000 images of handwritten digits (0-9).
- **12-Newsgroups**: A collection of approximately 11,308 documents, partitioned across 12 different newsgroups.

## Models

The models implemented in this project include:

- **Dense Autoencoder**: Used for synthetic datasets (e.g., Swiss Roll, S-Curve) - implemented in `dense_autoencoder.py`.
- **CNN Autoencoder**: Applied to the MNIST dataset - implemented in `cnn_autoencoder.py`.
- **Text Autoencoder**: Used for 12-Newsgroups data - implemented in `text_autoencoder.py`.
- **Geodesic Distance Constrained Autoencoder (GCAE)**: An advanced autoencoder that incorporates geodesic distance constraints to better capture the data's intrinsic geometry - implemented in `gcae.py`.

## Example Usage

To use the Geodesic Distance Constrained Autoencoder (GCAE) with a specific dataset, you can use the following example:

```python
# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from data.synthetic_data import load_synthetic_data
from models.dense_autoencoder import get_dense_autoencoder
from models.gcae import GCAETrainer

# Set seed
seed = 42

# Load dataset
X_train, X_test, y_train, y_test = load_synthetic_data('swiss_roll', n_samples=4000, noise=0.1, test_size=0.2, seed=seed)

# Load the autoencoder model
input_dim = 3
encoded_dim = 2 
dataset = "swiss_roll"
dense_autoencoder = get_dense_autoencoder(input_dim, encoded_dim, dataset)

# Instantiate the GCAETrainer with the dense autoencoder
trainer = GCAETrainer(
    model_type="dense",
    dataset=dataset,
    input_dim=input_dim,
    embedding_dim=encoded_dim,
    n_neighbors=7,
    alpha=10,
    seed=5
)

# Generate Geodesic distance matrix for train and validation data
train_geo_dist = trainer.compute_geodesic_distances(X_train, training=True)
val_geo_dist = trainer.compute_geodesic_distances(X_test, training=False)

# Convert data to TensorFlow Dataset
batch_size = 16
train_dataset = tf.data.Dataset.from_tensor_slices((X_train.astype(np.float32), np.arange(len(X_train))))
train_dataset = train_dataset.batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_test.astype(np.float32), np.arange(len(X_test))))
val_dataset = val_dataset.batch(batch_size)

# Create an EarlyStopping callback to terminate training if the validation total loss doesn't immprove after 10 epochs
early_stopping = EarlyStopping(monitor='val_total_loss', patience=10, mode='min', restore_best_weights=True)

# Compile and train the model
trainer.compile(optimizer='adam')
history = trainer.fit(train_dataset, epochs=100, validation_data=val_dataset, callbacks=early_stopping)

# Print final training and validation loss (MSE)
final_train_loss = history.history['total_loss'][-1]
final_val_loss = history.history['val_total_loss'][-1]
print(f"Final Training Loss: {final_train_loss:.4f}, Final Validation Loss: {final_val_loss:.4f}")

# Examine the training and validation loss over epochs
plt.figure(figsize=(20,4))
plt.subplot(1,3,1)
plt.plot(history.history['total_loss'], label='train')
plt.plot(history.history['val_total_loss'], label='val')
plt.xlabel("Epoch")
plt.ylabel("Total Loss (MSE + Geodesic Loss)")
plt.title("Total Loss vs epoch")
plt.legend()

plt.subplot(1,3,2)
plt.plot(history.history['reconstruction_loss'], label='train')
plt.plot(history.history['val_reconstruction_loss'], label='val')
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Reconstruction Loss vs epoch")
plt.legend()

plt.subplot(1,3,3)
plt.plot(history.history['geodesic_loss'], label='train')
plt.plot(history.history['val_geodesic_loss'], label='val')
plt.xlabel("Epoch")
plt.ylabel("Geodesic Loss")
plt.title("Geodesic Loss vs epoch")
plt.legend()
plt.show()
```

## Dependencies
This project was developed using Python 3.11.9. Additionally, the following Python libraries are required:
- faiss-cpu	1.8.0
- matplotlib 3.7.3
- memory-profiler 0.61.0
- nltk	3.8.1
- numba	0.60.0
- numpy	1.26.4
- scikit-learn	1.5.0
- scipy	1.13.1
- sentence-transformers	3.0.1
- tensorflow 2.14.1
- tensorflow-datasets 4.9.4
- torch	2.4.0

These can be installed via pip using the command `pip install faiss-cpu==1.8.0 matplotlib==3.7.3 memory-profiler==0.61.0 nltk==3.8.1 numba==0.60.0 numpy==1.26.4 scikit-learn==1.5.0 scipy==1.13.1 sentence-transformers==3.0.1 tensorflow==2.14.1 tensorflow-datasets==4.9.4 torch==2.4.0`.

## Code License

Currently, the code in this repository is not under any specific open-source license and is shared for educational and research purposes.
