from utils.custom_warning_ignore import IgnoreWarning
IgnoreWarning()
import os
import numpy as np
import json
import sys
import nibabel as nib
import matplotlib.pyplot as plt
import SimpleITK
import pandas as pd

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LambdaCallback

from utils import metrics
from generator.train_generator import Generator
from model.model_unet_3d import unet3d
from utils.dataio import import_data_filename, write_nii

import sys

#TASK 1: Understand the flow of the code
#       Add Comments wherever necessary to make sure you understand the code
#       Change the patch_size based on the images
#       Understand how loss function works. Try changing the loss function and see how the performance changes.
#       Change the epoch size & batch size and see how it is affecting the memory and performance.
#       Change the number of encoding/decoding layers and see if it changes the performance. 
#       Perform a dry run to ensure the code is running from end to end. NOTE: There will definitely be bugs when you start the project. But it will help you to understand the code better
#       Perform code refraction wherever necessary 
class TrainModel:
    def __init__(self):
        self.distributed = tf.distribute.MirroredStrategy()
        with open('./configs/config.json') as cf:
            self.config=json.load(cf)
        print(self.config)
        self.train_dir = self.config['train']['train_image_folder']
        self.valid_dir = self.config['valid']['valid_image_folder']
        self.test_dir = self.config['valid']['valid_image_folder']
        self.target_size = self.config['model']['target_size']
        self.n_channels = self.config['model']['n_channels']
        self.type =self.config['model']['type']
        self.labels =self.config['model']['labels']
        self.patch_size = [self.target_size, self.target_size, self.target_size]
        self.train_val_split=self.config['model']['train_val_split']
        self.model_name=self.config['train']['saved_weights_name']
        self.loss_function=self.config['model']['loss_function']

        self.augmentation = self.config['model']['augmentation']
        self.learning_rate = self.config['model']['learning_rate']
        self.epochs = self.config['model']['epochs']
        self.steps_per_epoch = self.config['model']['steps_per_epoch']
        self.load_weights = self.config['model']['load_weights']
        self.validation_steps = self.config['valid']['valid_times']
        self.load_weights_name = self.config['model']['load_weights_name']
        self.subject_list=pd.read_csv(self.config['model']['list'])
        self.best_val_acc = 0
        self.best_val_loss = sys.float_info.max

    def load_data_generator(self):
        self.n_training = int(len(self.subject_list)*0.6)
        self.n_validation = len(self.subject_list)-int(len(self.subject_list)*self.train_val_split)
        print("Number of training:" + str(self.n_training))
        print("Number of testing:" + str(self.n_validation))
        print(self.augmentation, self.model_name)

        self.training_generator = Generator(self.subject_list[:self.n_training], batch_size=10, patch_size=self.patch_size, labels=self.labels, augmentation=self.augmentation)
        self.validation_generator = Generator(self.subject_list[self.n_training:], batch_size=4, patch_size=self.patch_size, labels=self.labels, augmentation=self.augmentation)
        train_tf_gen = tf.data.Dataset.from_generator(self.training_generator.get_item, (tf.float32, tf.int16), (tf.TensorShape([self.patch_size[0], self.patch_size[1], self.patch_size[2], 1]), tf.TensorShape([self.patch_size[0], self.patch_size[1], self.patch_size[2], len(self.labels)])))
        self.train_batches = train_tf_gen.batch(12)

        valid_tf_gen = tf.data.Dataset.from_generator(self.validation_generator.get_item, (tf.float32, tf.int16), (tf.TensorShape([self.patch_size[0], self.patch_size[1], self.patch_size[2], 1]), tf.TensorShape([self.patch_size[0], self.patch_size[1], self.patch_size[2], len(self.labels)])))
        self.valid_batches = valid_tf_gen.batch(12)

    def saveModel(self, epoch, logs):
        val_acc = logs['val_dice_multi']
        val_loss = logs['val_loss']

        if val_acc > self.best_val_acc:
            print("Saving model. Loss improved from %s to %s and accuracy improved from %s to %s" %(self.best_val_loss, val_loss, self.best_val_acc, val_acc))
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.model.save_weights(self.model_name)


    def train(self):
        try:
            self.loss_function = getattr(metrics, self.loss_function)
        except:
            self.loss_function = self.loss_function

        with self.distributed.scope():
            self.model = unet3d(self.patch_size+[1], len(self.labels))
            optimizer = optimizers.Adam(lr=self.learning_rate)
            self.model.compile(optimizer=optimizer,
                      loss=self.loss_function,
                      metrics=['acc',metrics.dice_multi])

            check_pointer = ModelCheckpoint(filepath=self.model_name, verbose=1, save_best_only=True, save_weights_only=True)
            tensorboard = TensorBoard(log_dir='./graph')

            try:
                if self.load_weights=="True":
                    self.model.load_weights(self.load_weights_name)
                print("Successfully loaded weights")
            except:
                print("Unable to load weights")
        self.model.summary()
        self.model.fit(self.train_batches,
                    steps_per_epoch=self.steps_per_epoch,
                    epochs=self.epochs,
                    validation_data=self.valid_batches,
                    validation_steps=self.validation_steps,
                    callbacks=[check_pointer, tensorboard],
                    workers=10)


if __name__=='__main__':
    trainmodel = TrainModel()
    trainmodel.load_data_generator()
    trainmodel.train()
