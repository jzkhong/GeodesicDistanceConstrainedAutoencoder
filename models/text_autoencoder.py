import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import GlorotUniform

def get_text_encoder(input_dim, encoded_dim, dataset, seed=7):
    inputs = Input(shape=(input_dim,))
    x = Dense(1024, activation='elu', kernel_initializer=GlorotUniform(seed=seed))(inputs)
    x = Dense(512, activation='elu', kernel_initializer=GlorotUniform(seed=seed))(x)
    x = Dense(256, activation='elu', kernel_initializer=GlorotUniform(seed=seed))(x)
    encoded = Dense(encoded_dim, kernel_initializer=GlorotUniform(seed=seed))(x)
    return Model(inputs, encoded, name=f'{dataset}_encoder')

def get_text_decoder(input_dim, encoded_dim, dataset, seed=7):
    encoded_inputs = Input(shape=(encoded_dim,))
    x = Dense(256, activation='elu', kernel_initializer=GlorotUniform(seed=seed))(encoded_inputs)
    x = Dense(512, activation='elu', kernel_initializer=GlorotUniform(seed=seed))(x)
    x = Dense(1024, activation='elu', kernel_initializer=GlorotUniform(seed=seed))(x)
    decoded = Dense(input_dim, kernel_initializer=GlorotUniform(seed=seed))(x)
    return Model(encoded_inputs, decoded, name=f'{dataset}_decoder')

def get_text_autoencoder(input_dim, encoded_dim, dataset, seed=7):
    encoder = get_text_encoder(input_dim, encoded_dim, dataset, seed=seed)
    decoder = get_text_decoder(input_dim, encoded_dim, dataset, seed=seed)
    inputs = Input(shape=(input_dim,))
    encoded = encoder(inputs)
    decoded = decoder(encoded)
    return Model(inputs, decoded, name=f'{dataset}_autoencoder')
