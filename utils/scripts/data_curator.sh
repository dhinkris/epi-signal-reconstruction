#!/bin/bash
path='/home/dhinesh/Desktop/brain_detection/fetal_yolo/create_dataset/scripts/annotation_utils'
$path/nifty2jpg.sh
/Users/Shared/anaconda/bin/python3 $path/createAnnotations.py
$path/createXML.sh
