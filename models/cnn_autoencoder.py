import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Flatten, Dense, Reshape, Input
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform

def get_cnn_encoder(input_shape, encoded_dim, dataset, seed=7):
    inputs = Input(shape=input_shape)
    x = Conv2D(32, 3, activation='relu', strides=2, padding='same', kernel_initializer=GlorotUniform(seed=seed))(inputs)
    x = Conv2D(64, 3, activation='relu', strides=2, padding='same', kernel_initializer=GlorotUniform(seed=seed))(x)
    x = Flatten()(x)
    encoded = Dense(encoded_dim, kernel_initializer=GlorotUniform(seed=seed))(x)
    return Model(inputs, encoded, name=f'{dataset}_cnn_encoder')

def get_cnn_decoder(input_shape, encoded_dim, dataset, seed=7):
    encoded_inputs = Input(shape=(encoded_dim,))
    x = Dense(7*7*64, activation='relu', kernel_initializer=GlorotUniform(seed=seed))(encoded_inputs)
    x = Reshape((7, 7, 64))(x)
    x = Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same', kernel_initializer=GlorotUniform(seed=seed))(x)
    decoded = Conv2DTranspose(input_shape[-1], 3, activation='sigmoid', strides=2, padding='same', kernel_initializer=GlorotUniform(seed=seed))(x)
    return Model(encoded_inputs, decoded, name=f'{dataset}_cnn_decoder')

def get_cnn_autoencoder(input_shape, encoded_dim, dataset, seed=7):
    encoder = get_cnn_encoder(input_shape, encoded_dim, dataset, seed=seed)
    decoder = get_cnn_decoder(input_shape, encoded_dim, dataset, seed=seed)
    inputs = Input(shape=input_shape)
    encoded = encoder(inputs)
    decoded = decoder(encoded)
    return Model(inputs, decoded, name=f'{dataset}_cnn_autoencoder')
