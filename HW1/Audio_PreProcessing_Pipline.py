import os
import numpy as np
import time
from scipy.io import wavfile
from scipy import signal
import tensorflow as tf
from pathlib import Path
from numpy import linalg as LA
import pandas as pd
import argparse
import itertools
# from Utility import Change_frequency_edge_hertz , SNR , MFCC_FAST , MFCC_SLOW , Compute_SNR , Optimum
from subprocess import Popen
Popen('sudo sh -c "echo performance >''/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',shell=True).wait()

#####################################################           MFFC fUNCTIONS          ###########################################################################

"""input Parameteres :
Inputfoldername  
OutputFolderName 
length 
stride 
num_mel_bins 
MFCC_Co 
num_mel_bins
sampling_rate
"""


#####################################################            Compute_SNR          ###########################################################################
def Compute_SNR(Inputfoldername, OutputFolderName, length, stride , MFCC, num_mel_bins, sampling_rate,lower_edge_hertz, upper_edge_hertz , debug ):
    i=0
    Total_Execution_SLOW=0
    Total_Execution_FAST=0
    Total_SNR=0
    linear_to_mel_weight_matrix_s = None
    linear_to_mel_weight_matrix_f = None
    compute = False 
    for filename in os.listdir(Inputfoldername):
    #unzip the file before doing this instruction
        
        
        if filename.endswith(".wav"):
            if i == 0 :        # to speed up the processing pipline excution time, the linear_to_mel_weight_matrix is computed once from the first record. 
                mfccs_slow, Time_slow , linear_to_mel_weight_matrix_s , mfccs_slow_shape =  MFCC_SLOW (Inputfoldername, filename, OutputFolderName ,compute = True , debug = debug) 
                mfccs_fast, Time_fast , linear_to_mel_weight_matrix_f , mfccs_fast_shape = MFCC_FAST(Inputfoldername, filename, OutputFolderName, length, stride , MFCC, num_mel_bins, sampling_rate , lower_edge_hertz, upper_edge_hertz, compute = True , debug = debug)
                SNR_final = SNR(mfccs_slow,mfccs_fast)
                linear_to_mel_weight_matrix_s = linear_to_mel_weight_matrix_s
                linear_to_mel_weight_matrix_f = linear_to_mel_weight_matrix_f
            else :
                Slow, Time_slow ,_ , _ =  MFCC_SLOW (Inputfoldername, filename, OutputFolderName ,linear_to_mel_weight_matrix = linear_to_mel_weight_matrix_s  , debug = debug) 
                Fast, Time_fast,_, _ = MFCC_FAST(Inputfoldername, filename, OutputFolderName, length, stride , MFCC, num_mel_bins, sampling_rate , lower_edge_hertz, upper_edge_hertz,linear_to_mel_weight_matrix = linear_to_mel_weight_matrix_f, compute = False, debug = debug )
                SNR_final = SNR(mfccs_slow,mfccs_fast)
            print('\r',i,end='')    
            Total_Execution_SLOW += Time_slow
            Total_Execution_FAST += Time_fast
            Total_SNR += SNR_final
            i+=1
        if i == 100 and debug == True :
            print (f"Debuging mode analyze {i} files ")
            break
        Average_Execution_SLOW= (Total_Execution_SLOW/i)*1000
        Average_Execution_FAST=(Total_Execution_FAST/i)*1000
        Average_SNR=Total_SNR/i 
    
#####################################################           Printing the Results          ###########################################################################
    # print the shape of  the matrix mfccs_slow_shape  
    if Average_Execution_FAST <= 18.5 :                            ############ we set the constraint of excution time 
        print(f'the shape of  the matrix mfccs_slow_shape :{mfccs_slow_shape} ')  
        # print the shape of  the matrix mfccs_fast_shape  
        print(f'the shape of  the matrix mfccs_fast_shape :{mfccs_fast_shape} ') 
        
        # Relative time excution reducton by the fast pipline with respect to slow    
        print(f'The fast pipeline has :{100*(Average_Execution_SLOW - Average_Execution_FAST)/Average_Execution_SLOW :0.2f} % lower execution time compared to slow pipeline')
        #Average time of SLOW
        print(f'Average time of SLOW:{Average_Execution_SLOW} ms')
        #Average time of FAST
        print(f'Average time of FAST:{Average_Execution_FAST} ms')
        #Average value of the SNR
        print(f'Average value of the SNR:{Average_SNR} dB')


    return Average_SNR, Average_Execution_FAST, Average_Execution_SLOW
#####################################################            MFCC_SLOW          ###########################################################################

def MFCC_SLOW (Inputfoldername, filename, OutputFolderName , debug, linear_to_mel_weight_matrix = None , compute = False ) :
    
    #Fixed variables
    length= 0.016
    stride= 0.008
    num_mel_bins= 40
    MFCC= 10
    num_mel_bins = 40   
    sampling_rate = 16000 
    
    #print('==============================================')
    inputpath = f'./{Inputfoldername}/{filename}'
    start = time.time()
    audio = tf.io.read_file(inputpath)


    
    #START TIME
    start = time.time()
      
    tf_audio, rate = tf.audio.decode_wav(audio)
    # decode to tf tensor it's a design choice but it's better optimized by TensorFlow.
    tf_audio = tf.squeeze(tf_audio, 1)
    #  needs also the number of Input channels = number of used microphones in our case it's only = 1 
    # Remember : f.signal.stft takes frame_length, frame_step as Number of samples Not as in seconds so we need to apply the following 
    frame_length = int(length * rate.numpy())   # (args.length [in seconds] * rate.numpy() [in Hz 1/s]) = sec*1/s => Number of samples
    frame_step = int(stride * rate.numpy())     # (args.Stride [in seconds] * rate.numpy() [in Hz 1/s]) = sec*1/s => Number of samples
    #print('Frame length:', frame_length)
    #print('Frame step:', frame_step)
    stft = tf.signal.stft(tf_audio, frame_length ,  frame_step , fft_length=frame_length)

    spectrogram = tf.abs(stft)                         #  the output is a complex number and we are only interested in the magnitude (the real part) we should apply absolute |stft|
    # print('Spectrogram shape:', spectrogram)    # Spectogram is 2D tensor Time vs Frequency 
    del stft

    if compute == True :
        linear_to_mel_weight_matrix_s = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, spectrogram.shape[-1], sampling_rate, 20, 4000)  
    else :
        linear_to_mel_weight_matrix_s = linear_to_mel_weight_matrix
    # print('linear matrix shape:', linear_to_mel_weight_matrix)
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix_s, 1)          # this depends on the input audio file


    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix_s.shape[-1:]))
###############

    mfccs_slow = tf.signal.mfccs_from_log_mel_spectrograms(tf.math.log(mel_spectrogram + 1.e-6))[..., :MFCC] 
    del mel_spectrogram
    # print(f"mfccs_slow shape : {mfccs_slow.shape}")
    mfccs_byte = tf.io.serialize_tensor(mfccs_slow)
    mfccs_slow_shape = mfccs_slow.shape
    #Execution time
    end = time.time()
    
########################################################### Printing the Results if not in Debug mode 
    # Check if the Output Folder already exists if not make a new directory 
    if debug == False :
        p = Path(OutputFolderName)

        if p.exists() == False : 
            os.makedirs(OutputFolderName)

        filename_byte = f'./{OutputFolderName}/{filename}_mfccs_slow.tf'
        tf.io.write_file(filename_byte, mfccs_byte) 
        #end = time.time()
        #print('MFCCs shape:', mfccs.shape)  
          
    execution_time = end - start                                 ####### Compute the excution Time 
    

    return mfccs_slow, execution_time , linear_to_mel_weight_matrix_s , mfccs_slow_shape


#####################################################            MFCC_FAST          ###########################################################################
def MFCC_FAST(Inputfoldername, filename, OutputFolderName, length, stride , MFCC, num_mel_bins, sampling_rate , lower_edge_hertz, upper_edge_hertz , debug, linear_to_mel_weight_matrix = None , compute = False) :
    
    #print('==============================================')
    inputpath = f'./{Inputfoldername}/{filename}'
    start = time.time()
 
    input_rate, audio = wavfile.read(f'./yes_no/{filename}')
    audio = tf.io.read_file(inputpath)
    #START TIME
    start = time.time()
    

    
    tf_audio, rate = tf.audio.decode_wav(audio)    # The -32768 to 32767 signed 16-bit values will be scaled to -1.0 to 1.0 in float.
    # decode to tf tensor it's a design choice but it's better optimized  
    tf_audio = tf.squeeze(tf_audio, 1)        
    # f.signal.stft takes frame_length, frame_step as Number of samples Not as in seconds so we need to apply the following 
    frame_length = int(length * rate.numpy())   # (args.length [in seconds] * rate.numpy() [in Hz 1/s]) = sec*1/s => Number of samples
    frame_step = int(stride * rate.numpy())     # (args.Stride [in seconds] * rate.numpy() [in Hz 1/s]) = sec*1/s => Number of samples
    #print('Frame length:', frame_length)
    #print('Frame step:', frame_step)
    
    stft = tf.signal.stft(tf_audio, frame_length ,  frame_step , fft_length=frame_length) #Takes signl in form of A [..., samples] float32/float64 Tensor of real-valued signals.

    #print('Execution Time: {:.4f}s'.format(end-start))
    spectrogram = tf.abs(stft)                         #  the output is a complex number and we are only interested in the magnitude (the real part), we apply absolute |stft|
    del stft 
##################################     
    if compute == True :
        linear_to_mel_weight_matrix_f = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, spectrogram.shape[-1], sampling_rate, lower_edge_hertz, upper_edge_hertz) 
        
    else :
        linear_to_mel_weight_matrix_f = linear_to_mel_weight_matrix
    # print('linear shape:', linear_to_mel_weight_matrix)
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix_f, 1)          # this depends on the input audio file shape


    mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix_f.shape[-1:]))
#######################

    mfccs_fast = tf.signal.mfccs_from_log_mel_spectrograms(tf.math.log(mel_spectrogram + 1.e-6))[..., :MFCC] 
    
    mfccs_fast_shape = mfccs_fast.shape
    del  mel_spectrogram
    # print(f"mfccs_fast shape : {mfccs_fast.shape}")
    mfccs_byte = tf.io.serialize_tensor(mfccs_fast)

      
    #Execution time
    end = time.time()


    
########################################################### Printing the Results if not ib Debug mode 
    # Check if the Output Folder already exists if not make a new directory 
    if debug == False :
        p = Path(OutputFolderName)

        if p.exists() == False : 
            os.makedirs(OutputFolderName)

        filename_byte = f'./{OutputFolderName}/{filename}_mfccs_slow.tf'
        tf.io.write_file(filename_byte, mfccs_byte) 
        #end = time.time()
       # print('MFCCs shape:', mfccs.shape)  
          
    execution_time = end - start                                 ####### Compute the excution Time 
    
    return mfccs_fast, execution_time , linear_to_mel_weight_matrix_f , mfccs_fast_shape

########################################################### Compute SNR Function ###########################################################

def SNR(slow,fast):
    Norm_slow=LA.norm(slow)
    denom=LA.norm(slow-fast+(10**-6))
    snr = 20 * np.log10(Norm_slow / denom)
    return snr 

########################################################### This Function is for Parameter Tuning ONly ###########################################################
'''
def Change_frequency_edge_hertz (Inputfoldername , OutputFolderName , debug) :
    Inputfoldername = Inputfoldername
    OutputFolderName= OutputFolderName
    debug = debug 
    length= 0.016
    stride = 0.008
    MFCC= 10                   ###### Remember this changes the shape of the final output matrix
    num_mel_bins= 32
    sampling_rate= 16000
    lower_edge_hertz = np.arange(10,101,10) # Quality no effect on Excution time
    upper_edge_hertz = np.arange(200,4000,200)         ####### Remember upper should be < sampleRate/2 

 #####################################################################################################

    data={'lower_edge_hertz [Hz]':[], 'upper_edge_hertz [Hz]':[],  'Average_Execution_SLOW [ms]':[],'Average_Execution_FAST [ms]':[],'Average_SNR [db]':[] }

    for lower , upper in list(itertools.product(lower_edge_hertz, upper_edge_hertz)):
        print(f" compute for combination ({lower} , {upper}) ")
        lower_edge_hertz = lower
        upper_edge_hertz = upper
        Average_SNR,Average_Execution_FAST,Average_Execution_SLOW = Compute_SNR(Inputfoldername, OutputFolderName, length, stride , MFCC,num_mel_bins,  sampling_rate,lower_edge_hertz, upper_edge_hertz , debug)
        
        if Average_Execution_FAST <= 18 :
		data['lower_edge_hertz [Hz]'].append(lower_edge_hertz)
		data['upper_edge_hertz [Hz]'].append(upper_edge_hertz)
		data['Average_Execution_SLOW [ms]'].append(Average_Execution_SLOW)
		data['Average_Execution_FAST [ms]'].append(Average_Execution_FAST)
		data['Average_SNR [db]'].append(Average_SNR)
		
        print('#' * 100)  

    print(data) #####################################################################################################
'''
########################################################################### main function ###########################################################################




#the input parameters for the command line
parser=argparse.ArgumentParser()

parser.add_argument('--Inputfoldername',  type=str, help='Input folder name contains the input files for the pipeline')
parser.add_argument('--OutputFolderName', type=str, help='Output folder name')
parser.add_argument('--debug', default=False, action= 'store_true', help='debug can be either False/True if True we will not pring the results and excute the code for only 100 files')

args = parser.parse_args()

#parser info
Inputfoldername = f"./{args.Inputfoldername}"
OutputFolderName = f"./{args.OutputFolderName}"

debug = args.debug 

# print(f"Excute {operation}")
########################################################## Change_frequency_edge_hertz to recver from the dropping in quality due to reducing num num_mel_bins to 32
"""    
if operation == "frequency_edge" :
print(f"excuting frequency_edge")
Change_frequency_edge_hertz(Inputfoldername = Inputfoldername ,OutputFolderName = OutputFolderName , debug= debug )
"""
'''
length= 0.016
stride = 0.008
MFCC= 10                       ###### Remember this changes the shape of the final output matrix
num_mel_bins= 32
sampling_rate= 16000 
lower_edge_hertz = 400            # Quality no effect on Excution time
upper_edge_hertz = 2700          ####### Note : upper should be < sampleRate/2 
'''
Compute_SNR(Inputfoldername,OutputFolderName,length= 0.016,stride=0.008,MFCC= 10,num_mel_bins=32,sampling_rate= 16000,lower_edge_hertz=400,upper_edge_hertz=2700 , debug = debug)
