#!/bin/bash
path='/home/dhinesh/Desktop/brain_detection/fetal_yolo'
data=$path'/create_dataset/data/train.csv'
for subject in `cat $data`
do
	s=`echo $subject | cut -d ',' -f1`
	prefix=`echo $s | cut -d '.' -f1 | cut -d '/' -f11`
	echo $prefix
	mkdir -pv $path/dataset/training/images
	/Users/Shared/anaconda/bin/python3 /home/dhinesh/Desktop/brain_detection/med2image/bin/med2image -i $s -d $path/dataset/training/images -o $prefix".jpg"
done
