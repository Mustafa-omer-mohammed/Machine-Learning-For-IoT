import json
import base64
import numpy as np
import tensorflow as tf
import sys
import time 
import re
import os
import requests
from scipy import signal

# define the seed for both numpy and tensorflow
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
############################### Reading the testsplit and labels.txt ###############################
zip_path = tf.keras.utils.get_file(
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    fname='mini_speech_commands.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')

test_files = np.loadtxt("kws_test_split.txt" , dtype = str )
labels = np.loadtxt("labels.txt" , dtype = "object" ,delimiter= "," )

labels = [re.sub("[]''[]","", x) for x in labels]
labels = [re.sub("'","", x.strip()) for x in labels]
labels = np.array(labels , dtype = str) 
labels =  tf.convert_to_tensor(labels)






