from keras import layers as KLayers
from keras import Model
import tensorflow as tf

def CNN(input_shape, classes):
    X_input = KLayers.Input(input_shape)

    #convolution layer 1
    X = KLayers.Conv2D(32, 5, activation="relu", data_format="channels_last")(X_input)
    X = KLayers.BatchNormalization()(X)
    X = KLayers.MaxPooling2D(pool_size = (5,5), padding="same", data_format="channels_last")(X)
    X = KLayers.Dropout(0.4)(X)

    #convolution layer 2
    X = KLayers.Conv2D(64, 5, padding="Same", activation="relu", data_format="channels_last")(X)
    X = KLayers.BatchNormalization()(X)
    X = KLayers.MaxPooling2D(pool_size = (5,5), data_format="channels_last")(X)
    X = KLayers.Dropout(0.4)(X)

    X = KLayers.Flatten()(X)
    X = KLayers.Dense(64, activation = "relu", name="enc1")(X)
    X = KLayers.BatchNormalization(name="enc2")(X)
    X = KLayers.Dropout(0.2)(X)
    X = KLayers.Dense(classes, input_shape=input_shape, activation='softmax', name="enc3" + str(classes))(X)
    X = KLayers.Dropout(0.1)(X)
    model = Model(inputs = X_input, outputs = X, name='CNN')
    return model