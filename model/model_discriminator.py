import tensorflow as tf
from tensorflow.keras.layers import Dropout, LeakyReLU, Input, Activation, BatchNormalization, Concatenate, multiply, Flatten
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras import backend as K

def make_discriminator_model(input_shape=(100, 124, 1)):
    model = tf.keras.Sequential()
    model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=input_shape))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model