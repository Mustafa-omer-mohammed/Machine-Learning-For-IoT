import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow.lite as tflite
from tensorflow import keras
import zlib
import tensorflow_model_optimization as tfmot   

# Note : Python version used to excute the code is 3.7.11

#$$$$$$$$$$$$$

######################################################## Reading the data and split to Train , Validation and Test #########################################################

zip_path = tf.keras.utils.get_file(
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    fname='mini_speech_commands.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')

############## Using the splits provided by the text of the assignment 
train_files = tf.convert_to_tensor(np.loadtxt("kws_train_split.txt" , dtype = str ))
val_files = tf.convert_to_tensor(np.loadtxt("kws_val_split.txt" , dtype = str ) )
test_files = tf.convert_to_tensor(np.loadtxt("kws_test_split.txt" , dtype = str ))


# with silence ['stop', 'up', 'yes', 'right', 'left', 'no', 'silence', 'down', 'go']
LABELS = np.array(['stop', 'up', 'yes', 'right', 'left', 'no',  'down', 'go'] , dtype = str) 
print (f"The LABELS order as provided to the model are {LABELS}")

#$$$$$$$$$$$$$