import cherrypy
import json
import json
import base64
import numpy as np
import tensorflow as tf
import re
import os
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

def compute(  frame_length ,  num_mel_bins, sampling_rate, 
                    lower_frequency, upper_frequency):
    num_spectrogram_bins = (frame_length) // 2 + 1 
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, num_spectrogram_bins, sampling_rate,
                    lower_frequency, upper_frequency)
    return linear_to_mel_weight_matrix

############ Create the Keywords Spotting Class KWS ######################3
class  KWS(object):
    exposed = True
    def __init__(self):
        self.sampling_rate = 16000                              # 16000  
        self.frame_length = 640                                               # 640 
        self.frame_step = 320                                                   # 320 
        self.num_mel_bins = 40                                               # 40 
        self.lower_frequency = 20                                         # 20 
        self.upper_frequency = 4000                                         # 4000
        self.num_coefficients = 10 										# 10 
        num_spectrogram_bins = self.frame_length // 2 + 1

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
						self.num_mel_bins, num_spectrogram_bins, self.sampling_rate, 20, 4000)

    def preprocess(self ,audio_string):
        # decode and normalize
        audio_bytes = base64.b64decode(audio_string)
        audio, _ = tf.audio.decode_wav(audio_bytes)
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

    def GET(self, *path, **query):
        pass

    def POST(self, *path, **query):
        pass

    def PUT(self, *path, **query):

        pass
    def DELETE(self, *path, **query):
        pass


if __name__ == '__main__':
    conf = {'/': {'request.dispatch': cherrypy.dispatch.MethodDispatcher()}}
    cherrypy.tree.mount(KWS(), '', conf)
    cherrypy.config.update({'server.socket_host': '0.0.0.0'})
    cherrypy.config.update({'server.socket_port': 8080})
    cherrypy.engine.start()
    cherrypy.engine.block()
