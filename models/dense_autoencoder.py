import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform

def get_dense_encoder(input_dim, encoded_dim, dataset, seed=5):
    """
    Define the encoder part of an autoencoder.
    
    Parameters:
    input_dim (int): Dimension of the input data.
    encoded_dim (int): Dimension of the encoded representation.
    dataset (str): Name of the dataset.
    seed (int): Random seed for weight initialization.
    
    Returns:
    Model: Encoder model.
    """
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu', kernel_initializer=GlorotUniform(seed=seed))(inputs)
    x = Dense(64, activation='relu', kernel_initializer=GlorotUniform(seed=seed))(x)
    encoded = Dense(encoded_dim, kernel_initializer=GlorotUniform(seed=seed))(x)
    encoder = Model(inputs, encoded, name=f'{dataset}_encoder')
    return encoder

def get_dense_decoder(input_dim, encoded_dim, dataset, seed=5):
    """
    Define the decoder part of an autoencoder.
    
    Parameters:
    input_dim (int): Dimension of the output data.
    encoded_dim (int): Dimension of the encoded representation.
    dataset (str): Name of the dataset.
    seed (int): Random seed for weight initialization.
    
    Returns:
    Model: Decoder model.
    """
    encoded_inputs = Input(shape=(encoded_dim,))
    x = Dense(64, activation='relu', kernel_initializer=GlorotUniform(seed=seed))(encoded_inputs)
    x = Dense(128, activation='relu', kernel_initializer=GlorotUniform(seed=seed))(x)
    decoded = Dense(input_dim, kernel_initializer=GlorotUniform(seed=seed))(x)
    decoder = Model(encoded_inputs, decoded, name=f'{dataset}_decoder')
    return decoder
    
def get_dense_autoencoder(input_dim, encoded_dim, dataset, seed=5):
    """
    Construct the complete autoencoder by combining encoder and decoder.
    
    Parameters:
    input_dim (int): Dimension of the input data.
    encoded_dim (int): Dimension of the encoded representation.
    dataset (str): Name of the dataset.
    seed (int): Random seed for weight initialization.
    
    Returns:
    Model: Autoencoder model.
    """
    encoder = get_dense_encoder(input_dim, encoded_dim, dataset, seed=seed)
    decoder = get_dense_decoder(input_dim, encoded_dim, dataset, seed=seed)
    inputs = Input(shape=(input_dim,))
    encoded = encoder(inputs)
    decoded = decoder(encoded)
    autoencoder = Model(inputs=inputs, outputs=decoded, name=f'{dataset}_autoencoder')
    return autoencoder
