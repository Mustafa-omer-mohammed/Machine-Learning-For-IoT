
#Exercicio LAB2_exercicio1

import numpy as np
from scipy.io import wavfile
from scipy import signal
import os


rate, audio = wavfile.read('/home/pi/WORK_DIR/LAB2/test3.wav')

sampling_ratio= 48000;

audio = signal.resample_poly(audio, 1, sampling_ratio)

audio = audio.astype(np.int16)

wavfile.write('test_final.wav',sampling_ratio, audio);


size_original =  os.path.getsize('/home/pi/WORK_DIR/LAB2/test3.wav')
size_final =  os.path.getsize('/home/pi/WORK_DIR/LAB2/test_final.wav')

print("Original File: {} , Final File: {}, Prop: {:.2f}".format(size_original, size_final, size_original/size_final))

'''
ls -l

Original file   96044
Final file	46


'''
