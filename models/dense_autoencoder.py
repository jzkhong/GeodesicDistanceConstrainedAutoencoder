import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform

def get_dense_encoder(input_dim, encoded_dim, dataset, seed=5):
    inputs = Input(shape=(input_dim,))
    x = Dense(128, activation='relu', kernel_initializer=GlorotUniform(seed=seed))(inputs)
    x = Dense(64, activation='relu', kernel_initializer=GlorotUniform(seed=seed))(x)
    encoded = Dense(encoded_dim, kernel_initializer=GlorotUniform(seed=seed))(x)
    return Model(inputs, encoded, name=f'{dataset}_encoder')

def get_dense_decoder(input_dim, encoded_dim, dataset, seed=5):
    encoded_inputs = Input(shape=(encoded_dim,))
    x = Dense(64, activation='relu', kernel_initializer=GlorotUniform(seed=seed))(encoded_inputs)
    x = Dense(128, activation='relu', kernel_initializer=GlorotUniform(seed=seed))(x)
    decoded = Dense(input_dim, kernel_initializer=GlorotUniform(seed=seed))(x)
    return Model(encoded_inputs, decoded, name=f'{dataset}_decoder')

def get_dense_autoencoder(input_dim, encoded_dim, dataset, seed=5):
    encoder = get_dense_encoder(input_dim, encoded_dim, dataset, seed=seed)
    decoder = get_dense_decoder(input_dim, encoded_dim, dataset, seed=seed)
    inputs = Input(shape=(input_dim,))
    encoded = encoder(inputs)
    decoded = decoder(encoded)
    return Model(inputs, decoded, name=f'{dataset}_autoencoder')
