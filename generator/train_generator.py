from utils.custom_warning_ignore import IgnoreWarning
IgnoreWarning()

import numpy as np
from tensorflow import keras
import tensorflow as tf

from utils.one_hot_label import multi_class_labels
from utils.patch3d import patch
from utils.image_process import normlize_mean_std, crop_edge_pair

import os
import pandas as pd
import random
import nibabel as nib

from utils.transformation import (
    create_affine_matrix,
    similarity_transform_volumes,
)
from utils.augment import (add_gaussian_noise,
    add_speckle_noise,
    shot_noise,
    contrast_augment,
    apply_gaussian_filter)

def corr_matrix(file):
    data_set=open(file,'r')
    data_lst=[]
    for line in data_set:
        data_lst.append(line)
    labels=[]
    for x in range(0,200):
        labels.append(x)
    big_lst=[]
    time_series=[]
    for line in data_lst[5:]:
        temp_lst=line.split()
        np_lst=[]
        for elem in temp_lst:
            np_lst.append(float(elem))
        big_lst.append(np_lst)
    time_series=np.array(big_lst)

    return time_series

class Generator:
    'Generates data for Keras, based on array data X and Y'

    def __init__(self, subject_list, batch_size=32, image_height=None, image_width=None, labels=[1], augmentation=False):
        'Initialization'
        self.batch_size=batch_size
        self.augmentation = augmentation
        self.list_input_dir = subject_list
        self.labels = labels
        self.image_height=image_height
        self.image_width=image_width

        self._data=[]
        # subject_index = range(subject_list)
        print("Reading data...")
        for index, image in enumerate(subject_list['image']):
            if os.path.exists(image) :

                image_data = corr_matrix(image)

                #manipulate the label_data based on the region needs to be segmented
                if self.image_width<=image_data.shape[0] and self.image_height<=image_data.shape[1]:
                    _datadict=dict()
                    _datadict['_id'] = index
                    _datadict['image'] = image_data[:self.image_width, :]
                    _datadict['label'] = image_data[:self.image_width, :]
                    print(image_data[:self.image_width, :].shape)
                    self._data.append(_datadict)
        
        self.n_subject = len(self._data)
        print("Reading data completed {}".format(len(self._data)))
        self.on_epoch_end()

    def augment(self, data):
        #TASK 2
        pass

    def get_item(self):
        X = np.zeros((self.image_width, self.image_height, 1), dtype=np.float32)
        Y = np.zeros((self.image_width, self.image_height, 1), dtype=np.float32)  # channel_last by default

        while True:
            randomindex=random.randint(0,self.n_subject-1)
            selectedimage=self._data[randomindex]
            image_data = selectedimage['image']
            label_data = selectedimage['label']
            
            X[:,:,0] = image_data
            Y[:,:,0] = label_data

            yield X, Y



    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.batch_size)
        # np.random.shuffle(self.patch_indices)
        # self.indexes = np.arange(len(self.list_IDs))
