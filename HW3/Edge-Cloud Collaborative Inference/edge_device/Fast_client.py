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

if __name__ == '__main__':
	MFCC_OPTIONS = {'frame_length': 1024, 'frame_step': 310, 'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40, 'num_coefficients': 10}
	# MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320,   'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 16, 'num_coefficients': 10}
	first = True
	total_inference_time = 0
	i = 0
	slow = 0
	count = 0
	total = len(test_files)
	cost = 0
	for filename in test_files:
		print("*" * 100)
		print('  \r ',i,"\n",end='') 
		if first == True :
			print("\n computing")
			linear_to_mel_weight_matrix = compute( frame_length = 1024,  num_mel_bins = 40, sampling_rate = 16000, 
                    lower_frequency = 20, upper_frequency = 4000)
			first = False
		# print(f"first == {first}")
		kw_spotting = KWS(labels , filename ,linear_to_mel_weight_matrix, **MFCC_OPTIONS)
		predicted_label , check , best ,sec_best, audio_string,best , label_id , excution,label_t = kw_spotting.predict()
		print(f"\n Actual label is {label_id} , {label_t}")
		print(f"fast predicted label is {predicted_label }  probability {best*100 :0.2f}%  2nd prob ={sec_best*100 :0.2f}%  and diff = {check*100 :0.3f}% ")
		if best >=  0.49 and int(predicted_label) == label_id :
				count += 1
		if best < 0.49:
			# print(f"model predection is {soft_max} ,    {soft_max.sum()} \n")
			slow += 1
			print("Sending to the slow pipeline")

			url = 'http://192.168.43.99:8080/predict'  ### the notebook ip address
			# PACK INFO INTO A JSON
			to_predict = {
                        "bn": "raspberrypi.local",
                        "e": [{"n": "audio", "u": "/", "t": 0, "vd": audio_string}]}
			to_predict_senML_json = json.dumps(to_predict)
			
			size = sys.getsizeof(json.dumps(to_predict_senML_json))
			print(f"size = {size / 1048576} Mb")
			cost += size
			r = requests.put(url, json=to_predict)
			if r.status_code == 200:
				# print(r.text)
				rbody = r.json()
				# prob = rbody['prediction']
				slow_pred = rbody['prediction']
				if int(slow_pred) == label_id :
						count +=1
				print(f"The slow prediction is {slow_pred}")
			else:
				print("Error")
				print(r.text)
		
		
		
		print('Total inference {:.3f}ms'.format(excution))
		total_inference_time += excution

		i += 1
		# if i == 100 :
			# break
		time.sleep(1)
	print(f"i = {i} and total = {total}")
	# print(total_inference_time)
	avg_total_inference_time = total_inference_time / i 
	print(f"correct predictions = {count}")
	accuracy = count / i 

	print(f"accuracy = {accuracy * 100} % ")
	print(f"communication cost = {cost / 1048576} Mb and {slow} files sent to slow pipeline")
	print(f"The average Total inference time is {avg_total_inference_time} ms")




