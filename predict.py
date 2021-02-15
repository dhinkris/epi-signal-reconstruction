from utils.custom_warning_ignore import IgnoreWarning
IgnoreWarning()
import os
import numpy as np
import json
import nibabel as nib
import SimpleITK as sitk
import random
import time
import glob 

import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import tensorflow.keras.backend as K

from utils.image_process import normlize_mean_std, crop_edge_pair

from utils import metrics
from generator.predict_generator import PredictGenerator
from model.model_generator import unet3d
from model.model_discriminator import make_discriminator_model

from utils.patch3d import patch
from utils.one_hot_label import restore_labels
from utils.image_process import normlize_mean_std, crop_pad3D
import pandas as pd
import sys
# tf.get_logger().setLevel('ERROR')
K.set_learning_phase(0)

class Predictor:
    def __init__(self):
        with open('./configs/config_wholebrain_fmri.json') as cf:
            self.config=json.load(cf)
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

        self.generator = unet3d(self.patch_size+[1], 1)
        self.discriminator = make_discriminator_model(self.patch_size+[1])
        self.generator.summary()
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    def predict_gen(self, image_data, image_size):
        self.image_size = np.asarray(image_size)
        self.stride=[4, 4, 4]
        self.loc_patch = patch(image_size, self.patch_size, self.stride)
        output_size = np.append((self.loc_patch.size_after_pad),1)

        Y0 = np.zeros(output_size,  dtype=np.float32)

        odd = False
        n_patches = self.loc_patch.n_patch

        # if self.loc_patch.n_patch%2!=0:
        #     odd=True
        #     n_patches = self.loc_patch.n_patch-1

        # self.predict_generator = PredictGenerator(X, self.loc_patch,  batch_size=2, patch_size=self.patch_size, labels=self.labels, odd=odd)

        # predict_tf_gen = tf.data.Dataset.from_generator(self.predict_generator.get_item,
        #                                                                  (tf.float32),
        #                                                                  (tf.TensorShape([self.patch_size[0],
        #                                                                                   self.patch_size[1],
        #                                                                                   self.patch_size[2],
        #                                                                                   1])))

        # predict_tf_gen = predict_tf_gen.batch(2)

        checkpoint_dir = './training_checkpoints_wholebrain_fmri/'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                        discriminator_optimizer=self.discriminator_optimizer,
                                        generator=self.generator,
                                        discriminator=self.discriminator)

        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        prediction=[]
        for i in range(n_patches-1):
            each_patch=self.loc_patch.__get_single_patch__(image_data, i)
            each_patch=np.expand_dims(each_patch, axis=0)
            each_patch=np.expand_dims(each_patch, axis=-1)
            each_patch=normlize_mean_std(each_patch)
            prediction=self.generator(each_patch, training=True)
            # prediction.append(_pred)

            print(np.sum(prediction), np.min(prediction), np.max(prediction))

        # prediction=np.array(prediction)
        # print(prediction.shape)
        # print(Y0.shape)
        # for selected_patch in range(n_patches-1):
            # print(selected_patch)
            Y0 = self.loc_patch.__put_single_patch__(Y0, prediction, i)

        # if self.loc_patch.n_patch%2!=0:
        #     Xn = np.zeros([2]+self.patch_size+[1])
        #     image = self.loc_patch.__get_single_patch__(X, -1)
        #     image = normlize_mean_std(image)
        #     Xn[0,:,:,:,0] = image
        #     Xn[1,:,:,:,0] = np.zeros(self.patch_size)
        #     prediction=self.model.predict(Xn, workers=1)
        #     Y0 = self.loc_patch.__put_single_patch__(Y0, np.squeeze(prediction[0]), -1)

        Y = restore_labels(Y0, self.labels)
        result = crop_pad3D(Y, self.image_size)
        print(result.shape)
        return result


if __name__=='__main__':
    # input_dir='/data/mril/users/all/mrdata/research/processed/CNMC/chd_r01/fetal/wip/rs/fetuses/projects/stress/cerebellum/allNB/newborn_anat_files'
    p=Predictor()
    image_to_predict='/home/dhinesh/Desktop/generative-adverserial-networks/Fetal-Brain-Segmentation-3D-UNET/data/fetus_00750_minvol.nii.gz'
    nib_img=nib.load(image_to_predict)
    data=nib_img.get_fdata()
    if len(data.shape)==4:
        data=data[:,:,:,0]

    res = p.predict_gen(data, data.shape)
    nifti_img = nib.Nifti1Image(res, nib_img.affine)
    nib.save(nifti_img, os.path.join('./results/GAN_ITER1/', os.path.basename(image_to_predict).split('.')[0]+'_final.nii.gz'))
