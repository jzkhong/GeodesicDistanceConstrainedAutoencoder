import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Reshape, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform

def get_cnn_encoder(input_shape, encoded_dim, dataset, seed=7):
    """
    Define the CNN encoder part of a CNN autoencoder.
    
    Parameters:
    input_shape (tuple): Shape of the input data.
    encoded_dim (int): Dimension of the encoded representation.
    dataset (str): Name of the dataset.
    seed (int): Random seed for weight initialization.
    
    Returns:
    Model: CNN encoder model.
    """
    inputs = Input(shape=(input_shape))
    x = Conv2D(32, 3, activation='relu', strides=2, padding='same', 
               kernel_initializer=GlorotUniform(seed=seed))(inputs)
    x = Conv2D(64, 3, activation='relu', strides=2, padding='same', 
               kernel_initializer=GlorotUniform(seed=seed))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu', kernel_initializer=GlorotUniform(seed=seed))(x)
    x = Dropout(0.2)(x)
    encoded = Dense(encoded_dim, kernel_initializer=GlorotUniform(seed=seed))(x)
    cnn_encoder = Model(inputs, encoded, name=f'{dataset}_cnn_encoder')
    return cnn_encoder

def get_cnn_decoder(input_shape, encoded_dim, dataset, seed=7):
    """
    Define the CNN decoder part of a CNN autoencoder.
    
    Parameters:
    input_shape (tuple): Shape of the output data.
    encoded_dim (int): Dimension of the encoded representation.
    dataset (str): Name of the dataset.
    seed (int): Random seed for weight initialization.
    
    Returns:
    Model: CNN decoder model.
    """
    encoded_inputs = Input(shape=(encoded_dim,))
    x = Dense(64, activation='relu', kernel_initializer=GlorotUniform(seed=seed))(encoded_inputs)
    x = Dense(7*7*64, activation='relu', kernel_initializer=GlorotUniform(seed=seed))(x)
    x = Reshape((7, 7, 64))(x)
    x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same', 
                        kernel_initializer=GlorotUniform(seed=seed))(x)
    decoded = Conv2DTranspose(input_shape[-1], 3, activation='sigmoid', strides=2, padding='same', 
                              kernel_initializer=GlorotUniform(seed=seed))(x)
    cnn_decoder = Model(encoded_inputs, decoded, name=f'{dataset}_cnn_decoder')
    return cnn_decoder
    
def get_cnn_autoencoder(input_shape, encoded_dim, dataset, seed=7):
    """
    Construct the complete CNN autoencoder by combining the CNN encoder and CNN decoder.
    
    Parameters:
    input_shape (tuple): Shape of the input data.
    encoded_dim (int): Dimension of the encoded representation.
    dataset (str): Name of the dataset.
    seed (int): Random seed for weight initialization.
    
    Returns:
    Model: CNN autoencoder model.
    """
    cnn_encoder = get_cnn_encoder(input_shape, encoded_dim, dataset, seed=seed)
    cnn_decoder = get_cnn_decoder(input_shape, encoded_dim, dataset, seed=seed)
    inputs = Input(shape=(input_shape))
    encoded = cnn_encoder(inputs)
    decoded = cnn_decoder(encoded)
    autoencoder = Model(inputs=inputs, outputs=decoded, name=f'{dataset}_cnn_autoencoder')
    return autoencoder