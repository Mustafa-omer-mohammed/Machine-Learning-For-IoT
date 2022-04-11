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

############################### define Utility Functions ###############################

# Resampling function
def res(audio, sampling_rate):        
    audio = signal.resample_poly(audio, 1, 16000 // sampling_rate)
    return np.array(audio, dtype = np.float32)

# Translation of the resampling function from a numpy function to a tensorflow function
def tf_function(audio, sampling_rate):
    audio = tf.numpy_function(res, [audio, sampling_rate], tf.float32)
    return audio
####### softmax implementation  in numpy #############
def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x
############ Success checker policy function ##################
def success_checker (probabilityes):
	probabilityes = np.squeeze(probabilityes , axis=0)
	sorted_pro = np.sort(probabilityes)[::-1]
	# print(sorted_pro)
	best , sec_best = sorted_pro[:2]
	# print(best ,sec_best )
	dif = best - sec_best
	return dif , best ,sec_best
############## compute the linear to weigh matrix function ##############
def compute(  frame_length ,  num_mel_bins, sampling_rate, 
                    lower_frequency, upper_frequency):
    num_spectrogram_bins = (frame_length) // 2 + 1 
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sampling_rate,
                    lower_frequency, upper_frequency)
    return linear_to_mel_weight_matrix


###### The Kyewards Spotting Class ############

class KWS(object):
	def __init__(self, labels, file_path , linear_to_mel_weight_matrix,frame_length, frame_step, 
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None):
			self.labels = labels
			self.file_path = file_path
			self.sampling_rate = 16000                                             # 16000  
			self.frame_length = frame_length                                               # 640 
			self.frame_step = frame_step                                                   # 320 
			self.num_mel_bins = num_mel_bins                                               # 40 
			self.lower_frequency = lower_frequency                                         # 20 
			self.upper_frequency = upper_frequency                                         # 4000
			self.num_coefficients = num_coefficients 										# 10 
			self.linear_to_mel_weight_matrix    = linear_to_mel_weight_matrix                                   
		



	def preprocess(self , audio_binary):
		# decode and normalize
		audio, _ = tf.audio.decode_wav(audio_binary)
		audio = tf.squeeze(audio, axis=1)
		# Padding for files with less than 16000 samples
		zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)   
		audio = tf.concat([audio, zero_padding], 0)
		audio.set_shape([self.sampling_rate])

		stft = tf.signal.stft(audio, frame_length=self.frame_length,frame_step=self.frame_step, fft_length=self.frame_length)
		spectrogram = tf.abs(stft)

		mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1)
		log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
		mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
		mfccs = mfccs[..., :self.num_coefficients]

		mfccs = tf.expand_dims(mfccs, -1)
		mfccs = tf.expand_dims(mfccs, 0)

		return mfccs

	def read(self):
		audio_binary = tf.io.read_file(self.file_path)
		parts = self.file_path.split("/")
		parts = [f"'{part}'" for part in parts]
		label = parts[-2] 
		label = label[1:-1]
		label_id = tf.argmax(label == self.labels)
		
		audio_bytes = bytearray(open(self.file_path,'rb').read())
		audio_base64bytes =  base64.b64encode(audio_bytes)
		audio_string = audio_base64bytes.decode()
		return   audio_string , label, int(label_id ) , audio_bytes , audio_binary


	def predict (self):
		audio_string , label_t, label_id  , audio_bytes , audio_binary = self.read()
		start = time.time()
		mfccs = self.preprocess(audio_binary)	
		# print('Preprocessing {:.3f}ms'.format(preprocessing))
		interpreter = tf.lite.Interpreter(model_path='./kws_dscnn_True.tflite') # get the selected model
		interpreter.allocate_tensors()
		input_details = interpreter.get_input_details()
		output_details = interpreter.get_output_details()
		interpreter.set_tensor(input_details[0]['index'], mfccs)
		interpreter.invoke()
		predicted = interpreter.get_tensor(output_details[0]['index'])
		end = time.time()
		soft_max = softmax(predicted)
		predicted_label = np.argmax(soft_max)
		# predicted_prob = np.max(soft_max) 
		
		excution = (end-start)*1e3
		check ,best , sec_best= success_checker(soft_max)
		return  predicted_label , check , best ,sec_best, audio_string,best , label_id ,excution , label_t





