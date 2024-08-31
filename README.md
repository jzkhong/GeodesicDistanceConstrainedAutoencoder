# Geodesic Distance Constrained Autoencoder (GCAE)

This repository contains the code implementation for the Geodesic Distance Constrained Autoencoder (GCAE) developed as part of a UK MSc research project. The GCAE is designed to incorporate geodesic distances within the latent space of autoencoders to preserve the intrinsic geometry of the data.

## Table of Contents
- [Geodesic Distance Constrained Autoencoder (GCAE)](#geodesic-distance-constrained-autoencoder-gcae)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Datasets](#datasets)
  - [Models](#models)
  - [Usage](#usage)
  - [Dependencies](#dependencies)
  - [Code License](#code-license)

## Introduction

The GCAE framework is implemented to explore the impact of incorporating geodesic distance constraints in autoencoder architectures. The models included in this repository are specifically tailored for various datasets, including synthetic data, MNIST, and the 12-Newsgroups text data.

## Datasets

The following datasets are used in this study, with corresponding data loading and preprocessing scripts available in the `datasets` folder:

- **Synthetic Data**: Includes 3D Swiss Roll and S-Curve datasets.
- **MNIST**: Handwritten digits dataset.
- **12-Newsgroups**: A collection of approximately 20,000 newsgroup documents, partitioned across 20 different newsgroups.

## Models

The models implemented in this project include:

- **Dense Autoencoder**: Used for synthetic datasets (e.g., Swiss Roll, S-Curve) - implemented in `dense_autoencoder.py`.
- **CNN Autoencoder**: Applied to the MNIST dataset - implemented in `cnn_autoencoder.py`.
- **Text Autoencoder**: Used for 12-Newsgroups data - implemented in `text_autoencoder.py`.
- **Geodesic Distance Constrained Autoencoder (GCAE)**: An advanced autoencoder that incorporates geodesic distance constraints to better capture the data's intrinsic geometry - implemented in `gcae.py`.

## Usage

To use the code in this repository, you can directly run the specific dataset and model scripts:

- For synthetic datasets:
  ```bash
  python datasets/synthetic_data.py
  python models/dense_autoencoder.py
  python models/gcae.py
  ```
- For MNIST datasets:
  ```bash
    python datasets/mnist.py
    python models/cnn_autoencoder.py
    python models/gcae.py
  ```

- For 12-Newsgroups datasets:
  ```bash
    python datasets/newsgroups.py
    python models/text_autoencoder.py
    python models/gcae.py
  ```

## Dependencies
To run this notebook, the following specific versions of Python libraries are required:
- faiss-cpu	1.8.0
- matplotlib	3.7.3
- nltk	3.8.1
- numba	0.60.0
- numpy	1.26.4
- scikit-learn	1.5.0
- scipy	1.13.1
- sentence-transformers	3.0.1
- tensorflow	2.14.1
- tensorflow-datasets	4.9.4
- torch	2.4.0

These can be installed via pip using the command `pip install faiss-cpu==1.8.0 matplotlib==3.7.3 nltk==3.8.1 numba==0.60.0 numpy==1.26.4 scikit-learn==1.5.0 scipy==1.13.1 sentence-transformers==3.0.1 tensorflow==2.14.1 tensorflow-datasets==4.9.4 torch==2.4.0`.

## Code License

Currently, the code in this repository is not under any specific open-source license and is shared for educational and research purposes.
