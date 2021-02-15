import warnings
import tensorflow as tf
import logging
import os
from warnings import simplefilter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"]='1'

class IgnoreWarning:
    def __init__(self):
        warnings.filterwarnings("ignore")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.warn("deprecated", DeprecationWarning)
            warnings.warn("warn_on_dtype", DeprecationWarning)
            warnings.warn("deprecated", FutureWarning)
        simplefilter(action='ignore', category=FutureWarning)
        # tf.get_logger().setLevel(logging.ERROR)
