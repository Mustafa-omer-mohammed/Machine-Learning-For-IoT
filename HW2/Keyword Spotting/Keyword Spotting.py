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
######################################################## Create the SignalGenerator #########################################################


class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate                                             # 16000  
        self.frame_length = frame_length                                               # 640 
        self.frame_step = frame_step                                                   # 320 
        self.num_mel_bins = num_mel_bins                                               # 40 
        self.lower_frequency = lower_frequency                                         # 20 
        self.upper_frequency = upper_frequency                                         # 4000
        self.num_coefficients = num_coefficients                                       # 10 
        num_spectrogram_bins = (frame_length) // 2 + 1                                  # ( frame size // 2 ) + 1 

   

        if mfcc is True:                                          # to speed up the preprocessing we need to compute the linear_to_mel_weight_matrix once so it will be a class argument 
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                    self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def read(self, file_path):
        parts = tf.strings.split(file_path,  "/")
        label = parts[-2]                                 
        label_id = tf.argmax(label == self.labels)        # extract the label ID (the integer mapping of the label)
        audio_binary = tf.io.read_file(file_path)         # reading the audio file in byte format
        audio, _ = tf.audio.decode_wav(audio_binary)      # decode a 16-bit PCM WAV file to a float tensor
        audio = tf.squeeze(audio, axis=1)

        return audio, label_id

    def pad(self, audio):
        # Padding for files with length less than 16000 samples
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)     # if the shape of the audio is already = 16000 (sampling rate) we will add nothing 

        # Concatenate audio with padding so that all audio clips will be of the same length
        audio = tf.concat([audio, zero_padding], 0)
        # Unify the shape to the sampling frequency (16000 , )
        audio.set_shape([self.sampling_rate])

        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)                         # expand_dims will not add or reduce elements in a tensor, it just changes the shape by adding 1 to dimensions for the batchs. 
    
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls = tf.data.experimental.AUTOTUNE) # parallel mapping exploiting the best number of parallel workers 
        ds = ds.batch(32)                                                                # create batches of 32 samples
        ds = ds.cache()                                                                  # cashe is used to avoid recomputing the previous preprocessing
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)                                  # applied to start reading the next batch from memory while prpcessing the current one
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds
######################################################## Generate Data set splits #########################################################

generator = SignalGenerator(LABELS, 16000, **options)
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)
