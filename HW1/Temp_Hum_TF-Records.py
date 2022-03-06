import tensorflow as tf
import pandas as pd
import numpy as np
import time, datetime
import argparse
import os
import sys

# Sensor DHT11 Maximum and Minimum Temp,Hum values ==> used for normalization 
Temp_MAX=50
Temp_MIN=0
Hum_Max=90
Hum_Min=20




#the input parameters for the command line
parser=argparse.ArgumentParser()

parser.add_argument('--normalize', default=False, action= 'store_true', help='normalization False/True')
parser.add_argument('--output', type=str, required=True,help='output filename')
parser.add_argument('--input', type= str,default='input', help= 'input filename')

args = parser.parse_args()

#parser info
filename=f"{args.input}.csv"          ##### input file name 
filename_OUT= args.output             ##### output file name 
normalization = args.normalize        ##### Normalization Flag (if True apply Normalization)


print(f"Normalization: {normalization}")

#this is reading the filename that contains the info
try:
    df = pd.read_csv(filename, header=None, names=['date', 'time', 'temp', 'hum'])
except FileNotFoundError:
    print(f"Input file '{filename}' does not exist. Shutting down...")
    sys.exit()
############################################################### Utilization Fuctions ################################################################################
# Define a function to apply Normalization if needed 
def normalize(df):
    t_max = 50
    t_min = 0
    h_max = 90
    h_min = 20
    df['temp'] = (df['temp']-t_min)/(t_max-t_min)
    df['hum'] = (df['hum']-h_min)/(h_max-h_min)
    
    return df
# Read time as posix_time and convert to int 
def posix_time (timestamp):
	ts = datetime.datetime.strptime(timestamp, '%d/%m/%Y,%H:%M:%S')
	posix_ts = int(time.mktime(ts.timetuple()))

	return posix_ts

# Fucntion to get the size of file before and after conversion 
def getSize(filename):
    st = os.stat(filename)
    return st.st_size
############### tf examples conversion function ######################   
def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):

  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
############################################################### Write to TF Records Fuctions ################################################################################


if(normalization == True):
	#changing the name of the output file for the normalization
	filename_OUT= filename_OUT + "_Normalized"
	df = normalize(df)

#DateTime preprocessing --> create another column with the data and time concat
# map time to posix time
df['timestamp'] = df['date'] +","+df['time']
df['posix_ts'] = df['timestamp'].map(lambda t : posix_time(t))


Date_Time = df['posix_ts']
Temp = df['temp']
Humid = df['hum']



with tf.io.TFRecordWriter(filename_OUT) as writer:

	for i in range(len(df)):
		
		if normalization==True:    ####### if normalize ==> apply normalization to the Temperature and Humidity input values 

			mapping = {'Date_time': _int64_feature(Date_Time[i]),
					   'Temperature' : _float_feature(Temp[i]),
				       'Humidity' : _float_feature(Humid[i])}
						
			example = tf.train.Example(features=tf.train.Features(feature=mapping))

			writer.write(example.SerializeToString())	
			# print(example)	
			
		else:	#  normalization will not be applied
			
			mapping = {'Date_time': _int64_feature(Date_Time[i]),
					   'Temperature' : _int64_feature(Temp[i]),
					   'Humidity' : _int64_feature(Humid[i])}
						
						
			example = tf.train.Example(features=tf.train.Features(feature=mapping))

			writer.write(example.SerializeToString())
	
			# print(example)


############################################################### Read from TF Records  Fuctions ################################################################################
# please uncomment to check the output can be read as expected 

'''
dataset = tf.data.TFRecordDataset(filename_OUT )#buffer_size=100)
print(dataset)


# Read the data back out.

#Normalized
def decode_fn_N(example):
	#print("!!!!!!!!!!!!!!!!!!!!!!!!!!!Normalized in the if")
	return tf.io.parse_single_example(
	      # Data
	      example,

	      # Schema
	      {"Date_time": tf.io.FixedLenFeature([], dtype=tf.int64,),
	       "Temperature": tf.io.FixedLenFeature([], dtype=tf.float32,),
	       "Humidity": tf.io.FixedLenFeature([], dtype=tf.float32,)}
	  )
	  
#Not normalized	  
def  decode_fn(example):
	return tf.io.parse_single_example(
	# Data
	example,

	# Schema
	{"Date_time": tf.io.FixedLenFeature([], dtype=tf.int64,),
	"Temperature": tf.io.FixedLenFeature([], dtype=tf.int64,),
	"Humidity": tf.io.FixedLenFeature([], dtype=tf.int64,)}
	)



if len(filename_OUT)==len(args.output):
	#Not normalized
	for batch in tf.data.TFRecordDataset([filename_OUT]).map(decode_fn):
		#print(batch)
		
		print("Date_time = {Date_time},  Temperature = {Temperature}, Humidity = {Humidity}".format(**batch))
						
		
else: 
	
	for batch in tf.data.TFRecordDataset([filename_OUT]).map(decode_fn_N):
		#print(batch)
		print("Date_time = {Date_time},  Temperature = {Temperature:.2f}, Humidity = {Humidity:.2f}".format(**batch))
		#print(f"Size of the {filename_OUT} is {size.st_size}")
'''	



print(f"Size of the {filename_OUT}.Tfrecord is {getSize(filename_OUT)} B ")
		
print(f"Size of the {filename}.csv is {getSize(filename)} B")
