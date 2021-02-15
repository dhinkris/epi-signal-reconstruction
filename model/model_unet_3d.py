from tensorflow.keras.layers import Conv3D, MaxPool3D, concatenate, Input, Dropout, PReLU, Conv3DTranspose, BatchNormalization, TimeDistributed, Conv2D, ConvLSTM2D
from tensorflow.keras.models import Model
import tensorflow as tf


def unet_core(x, filter_size=8, kernel_size=(3, 3, 3)):
    x = Conv3D(filters=filter_size,
               kernel_size=kernel_size,
               padding='same',
               kernel_initializer='he_normal',dtype=tf.float32)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv3D(filters=filter_size,
               kernel_size=kernel_size,
               padding='same',
               kernel_initializer='he_normal',dtype=tf.float32)(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    return x

def unet3d(patch_size, n_label):
    # with distributed.scope():
    input_layer = Input(shape=patch_size,dtype=tf.float32)
    d1 = unet_core(input_layer, filter_size=96, kernel_size=(3, 3, 3))
    l = MaxPool3D(strides=(2, 2, 2))(d1)
    d2 = unet_core(l, filter_size=192, kernel_size=(3, 3, 3))
    l = MaxPool3D(strides=(2, 2, 2))(d2)
    d3 = unet_core(l, filter_size=384, kernel_size=(3, 3, 3))
    l = MaxPool3D(strides=(2, 2, 2))(d3)

    b = unet_core(l, filter_size=768, kernel_size=(3, 3, 3))

    l = Conv3DTranspose(filters=384, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal',dtype=tf.float32)(b)
    l = concatenate([l, d3])
    u3 = unet_core(l, filter_size=384, kernel_size=(3, 3, 3))
    l = Conv3DTranspose(filters=192, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal',dtype=tf.float32)(u3)
    l = concatenate([l, d2])
    u2 = unet_core(l, filter_size=192, kernel_size=(3, 3, 3))
    l = Conv3DTranspose(filters=96, kernel_size=(2, 2, 2),  padding='same', strides=2, kernel_initializer='he_normal',dtype=tf.float32)(u2)
    l = concatenate([l, d1])
    u1 = unet_core(l, filter_size=96, kernel_size=(3, 3, 3))
    output_layer = Conv3D(filters=n_label, kernel_size=(1, 1, 1), activation='sigmoid')(u1)
    # output_layer = CRF(n_label)
    model = Model(input_layer, output_layer)
    return model
