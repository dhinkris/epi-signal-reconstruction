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
from model.model_generator import make_generator_model, make_generator_model
from model.model_discriminator import make_discriminator_model

from utils.dataio import import_data_filename, write_nii
from tensorflow.python.client import device_lib
import sys
import time
import datetime

def get_available_gpus():
    local_device_protos=device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type=='GPU']

print(get_available_gpus())

class TrainModel:
    def __init__(self):
        self.distributed = tf.distribute.MirroredStrategy()
        with open('./configs/config.json') as cf:
            self.config=json.load(cf)
        self.train_dir = self.config['train']['train_image_folder']
        self.valid_dir = self.config['valid']['valid_image_folder']
        self.test_dir = self.config['valid']['valid_image_folder']
        self.n_channels = self.config['model']['n_channels']
        self.type =self.config['model']['type']
        self.labels =self.config['model']['labels']
        self.image_width = self.config['model']['image_width']
        self.image_height = self.config['model']['image_height']
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

        self.generator = make_generator_model(input_shape=(self.image_width, self.image_height, 1))
        self.discriminator = make_discriminator_model(input_shape=(self.image_width, self.image_height, 1))
        self.generator.summary()
        self.discriminator.summary()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


    def load_data_generator(self):
        self.n_training = int(len(self.subject_list)*0.6)
        self.n_validation = len(self.subject_list)-int(len(self.subject_list)*self.train_val_split)
        print("Number of training:" + str(self.n_training))
        print("Number of testing:" + str(self.n_validation))
        print(self.augmentation, self.model_name)

        self.training_generator = Generator(self.subject_list[:self.n_training], batch_size=10, image_height=self.image_height, image_width=self.image_width, augmentation=self.augmentation)
        self.validation_generator = Generator(self.subject_list[self.n_training:], batch_size=4, image_height=self.image_height, image_width=self.image_width, augmentation=self.augmentation)
        train_tf_gen = tf.data.Dataset.from_generator(self.training_generator.get_item, (tf.float32, tf.float32), (tf.TensorShape([self.image_width, self.image_height, 1]), tf.TensorShape([self.image_width, self.image_height, 1])))
        self.train_batches = train_tf_gen.batch(6)

        valid_tf_gen = tf.data.Dataset.from_generator(self.validation_generator.get_item, (tf.float32, tf.float32), (tf.TensorShape([self.image_width, self.image_height, 1]), tf.TensorShape([self.image_width, self.image_height, 1])))
        self.valid_batches = valid_tf_gen.batch(6)
    
    # Notice the use of `tf.function`
    # This annotation causes the function to be "compiled".
    #@tf.function
    def train_step(self, dataset, epoch):
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        def discriminator_loss(self, real_output, fake_output):
            real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
            fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
            total_loss = real_loss + fake_loss
            return total_loss

        def generator_loss(self, fake_output):
            return self.cross_entropy(tf.ones_like(fake_output), fake_output)


        image, label = dataset
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(image, training=True)
            print(generated_images)
            real_output = self.discriminator(label, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        tf.print("Generator loss: {} Generator GAN loss: {} Generator L1 loss: {} Discriminatory loss: {}".format(gen_loss, disc_loss))
        with self.summary_writer.as_default():
            tf.summary.scalar('gen_total_loss', gen_loss, step=epoch)
            tf.summary.scalar('disc_loss', disc_loss, step=epoch)
    
    def train(self):
        log_dir="logs/"

        self.summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        checkpoint_dir = './training_checkpoints_wholebrain'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)

        sess = tf.compat.v1.Session()
        for epoch in range(self.epochs):
            start = time.time()

            # for image_batch in self.train_batches:
            for steps_per_epoch in range(self.steps_per_epoch):
                with sess.as_default():
                    self.train_step(self.train_batches.make_one_shot_iterator().get_next(), epoch)
            
            tf.print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

            if (epoch + 1) % 5 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)
            
        checkpoint.save(file_prefix=checkpoint_prefix)

if __name__ == "__main__":
    trainModel = TrainModel()
    trainModel.load_data_generator()
    trainModel.train()
