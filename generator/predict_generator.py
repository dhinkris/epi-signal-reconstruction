from utils.custom_warning_ignore import IgnoreWarning
IgnoreWarning()

import numpy as np
from tensorflow import keras
import tensorflow as tf

from utils.one_hot_label import multi_class_labels
from utils.patch3d import patch
from utils.image_process import normlize_mean_std
import os
import random


class PredictGenerator(keras.utils.Sequence):
    'Generates data for Keras, based on array data X and Y'

    def __init__(self, image_data, loc_patch, batch_size=32, patch_size=[32, 32, 32], labels=[1], odd=False):
        'Initialization'
        self.image_data=image_data
        self.patch_size = np.asarray(patch_size)
        self.loc_patch = loc_patch
        self.patch_index=[]
        self.odd = odd
        self.on_epoch_end()

    def get_item(self):
        X = np.zeros((self.patch_size[0], self.patch_size[1], self.patch_size[2], 1), dtype=np.float32)
        m = 0
        n_patches = self.loc_patch.n_patch
        if self.odd==True:
            n_patches=self.loc_patch.n_patch-1

        while m<n_patches:
            patch_index = random.randint(0, n_patches-1)
            self.patch_index.append(patch_index)
            image = self.loc_patch.__get_single_patch__(self.image_data, m)
            image = normlize_mean_std(image)
            X[:,:,:,0] = image
            m+=1
            yield X

    def on_epoch_end(self):
        return self.patch_index
