#! /usr/bin/env python

import nibabel as nib
import numpy as np
import pandas as pd
import os

path='/home/dhinesh/Desktop/brain_detection/fetal_yolo'

image_path_=path+'/dataset/training/images/'

folders=['train']

def get_box_points(img):
    spx = 0
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    for x in range(1, img.shape[0]):
        if np.sum(img[:, x]) > 0:
            if spx == 0:
                xmin = x
                spx = 1
            if np.sum(img[:, x+1]) == 0:
                xmax = x
                spx = 0
    spy = 0
    for y in range(1, img.shape[0]):
        if np.sum(img[y, :]) > 0:
            if spy == 0:
                ymin = y
                spy = 1
            if np.sum(img[y+1, :]) == 0:
                ymax = y
                spy = 0

    return xmin, ymin, xmax, ymax

for folder in folders:
    data=pd.DataFrame()
    points=[]
    mask=[]

    xmin=[]
    ymin=[]
    xmax=[]
    ymax=[]
    filename=[]
    xsize=[]
    ysize=[]
    label=[]

    nifti_data = pd.read_csv(os.path.join(path, path+'/create_dataset/data/'+folder+'.csv'))
    for image in nifti_data['mask']:
        nifti = nib.load(image)
        for each_slice in range(0, nifti.shape[2]):
            if each_slice<10:
                filename.append(image_path_+'/'+(image.split('_mask')[0]+'-frame000-slice00'+str(each_slice)+'.jpg').split('/')[10])
            else:
                filename.append(image_path_+'/'+(image.split('_mask')[0]+'-frame000-slice0'+str(each_slice)+'.jpg').split('/')[10])
            point = get_box_points(nifti.get_fdata()[:,:,each_slice])
            xmin.append(point[0])
            ymin.append(point[1])
            xmax.append(point[2])
            ymax.append(point[3])
            xsize.append(256)
            ysize.append(256)
            label.append('brain')
    data['images'] = filename
    data['xsize'] = xsize
    data['ysize'] = ysize

    data['xmin'] = xmin
    data['ymin'] = ymin
    data['xmax'] = xmax
    data['ymax'] = ymax

    data['class'] = label
    data.to_csv(path+'/create_dataset/data/datapoint_'+folder+'.csv', index=None)
