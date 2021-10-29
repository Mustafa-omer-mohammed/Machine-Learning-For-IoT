from board import D4
import adafruit_dht

import time
import pyaudio
import wave
import sys

form_1 = pyaudio.paInt16 # 16-bit resolution
chans = 1 # 1 channel
samp_rate = 44100 # 44.1kHz sampling rate
chunk = 1000 # 2^12 samples for buffer
record_secs = 2 # seconds to record
dev_index = 1 # device index found by p.get_device_info_by_index(ii)
wav_output_filename = 'test3.wav' # name of .wav file

audio = pyaudio.PyAudio() # create pyaudio instantiation

# create pyaudio stream
stream = audio.open(format = form_1,rate = samp_rate,channels = chans, \
                    input_device_index = dev_index,input = True, \
                    frames_per_buffer=chunk)
print("recording")
frames = []

start=time.time()
# loop through stream and append audio chunks to frame array
for ii in range(0,int((samp_rate/chunk)*record_secs)):
	
	data = stream.read(chunk)
	frames.append(data)

print("finished recording")
end=time.time()
time=end-start
print(f"time is {time}")
# stop the stream, close it, and terminate the pyaudio instantiation
stream.stop_stream()
stream.close()
audio.terminate()

# save the audio frames as .wav file
wavefile = wave.open(wav_output_filename,'wb')
wavefile.setnchannels(chans)
wavefile.setsampwidth(audio.get_sample_size(form_1))
wavefile.setframerate(samp_rate)
wavefile.writeframes(b''.join(frames))
wavefile.close()


'''
frequency 	48.0kHz		size 288 044
frequency 	44.1kHz		size 176 044

'''
