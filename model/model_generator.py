from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Conv2DTranspose, Reshape, ConvLSTM2D
from tensorflow.keras.models import Model
import tensorflow as tf


def make_generator_model(input_shape=(100, 100, 1)):
    model = tf.keras.Sequential()
    model.add(Dense(25*31*256, use_bias=False, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Reshape((25, 31, 256)))
    assert model.output_shape == (None, 25, 31, 256) # Note: None is the batch size

    model.add(Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 25, 31, 128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 50, 62, 64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 100, 124, 1)

    return model
