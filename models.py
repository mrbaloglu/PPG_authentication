from keras import layers as KLayers
from keras import Model
from keras.initializers import GlorotUniform
import tensorflow as tf
import keras

def conv_block(inputs, n_filters, window_size, dropout, block_name="conv_block"):
    
    with keras.name_scope(block_name):
        X = KLayers.Conv2D(n_filters, window_size, padding="Same", activation="relu", data_format="channels_last")(inputs)
        X = KLayers.BatchNormalization()(X)
        X = KLayers.MaxPooling2D(pool_size = (window_size, window_size), padding="same", data_format="channels_last")(X)
        X = KLayers.Dropout(dropout)(X)

    return X

def dense_block(inputs, n_dense_units, dropout, block_name="dense_block"):
    
    with keras.name_scope(block_name):
        X = KLayers.Dense(n_dense_units, activation = "relu")(inputs)
        X = KLayers.BatchNormalization()(X)
        X = KLayers.Dropout(dropout)(X)

    return X

def CNN(input_shape, classes, num_conv_filters, conv_window_sizes, num_dense_units):
    X_input = KLayers.Input(input_shape)

    assert len(num_conv_filters) == len(conv_window_sizes)

    for i, (nfilters, wsize) in enumerate(zip(num_conv_filters, conv_window_sizes)):
        if i == 0:
            X = conv_block(X_input, nfilters, wsize, dropout=0.35, block_name=f"conv_block_{i}")
        else:
            X = conv_block(X, nfilters, wsize, dropout=0.35, block_name=f"conv_block_{i}")

    X = KLayers.Flatten()(X)

    for i, n_units in enumerate(num_dense_units):
        X = dense_block(X, n_units, dropout=0.2, block_name=f"dense_block_{i}")

    X = KLayers.Dense(classes, activation='softmax', name="enc3" + str(classes))(X)
    model = Model(inputs = X_input, outputs = X, name='CNN')
    return model