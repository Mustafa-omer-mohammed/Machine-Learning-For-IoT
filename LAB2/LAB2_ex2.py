import argparse
import numpy as np
from scipy.io import wavfile
from scipy import signal
import os
import tensorflow as tf
import time

parser=argparse.ArgumentParser()
parser.add_argument('-f', type=float, help='frequency', required = True)
parser.add_argument('--length', type=float, help='frame lenght', required = True )
parser.add_argument('--stride', type=float, help='stride', required = True)
parser.add_argument('--filename', type=str, help='filename', required = True)

args = parser.parse_args()

#Reading the audio from exercicise 1
audio= tf.io.read_file('/home/pi/WORK_DIR/LAB2/test_final.wav')

tf_audio, rate = tf.audio.decode_wav(audio)
tf_audio = tf.squeeze(tf_audio, 1)
#tf_audio= tf.cast(tf_audio, tf.int32)

frame_length= int(args.length * rate.numpy())
frame_step= int(args.stride * rate.numpy())
print(f"Frame length: {frame_length}")
print(f"Frame step: {frame_step}")

start = time.time()
stft = tf.signal.stft(tf_audio, frame_length , frame_step ,fft_length=frame_length)

end= time.time()
duration= end-start
print(f"Duration: {duration}")

spectrogram = tf.abs(stft)

spectrogram_byte= tf.io.serialize_tensor(spectrogram)
filename_byte='{}.tf'.format(os.path.splitext(args.filename)[0])
tf.io.write_file(filename_byte, spectrogram_byte)
input_size= os.path.getsize(args.filename)/2.**10
spectrogram_size=os.path. getsize(filename_byte)/2.**10
print("Input size: {:.2f}KB.".format(input_size))
print(" Spectrogram size: {:.2f}KB".format(spectrogram_size))
