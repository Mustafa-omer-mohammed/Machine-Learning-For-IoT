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

with tf.io.TFRecordWriter(filename_OUT) as writer:

	if normalization==True:
		#Values according to normalization
		data["Temperature"] = (data["Temperature"] - Temp_MIN) / (Temp_MAX - Temp_MIN)
		data["Humidity"]    = (data["Humidity"] - Hum_Min) / (Hum_Max - Hum_Min)
		
		Temp=data.Temperature
		Humid=data.Humidity
		
		print("Norm done")
		print(data.head())
		
		for i in range(len(data)):
		
			Date_time = tf.train.Feature(float_list=tf.train.FloatList(value=[Date_Time[i]]))
			#print(Date_time)
			Temperature= tf.train.Feature(float_list=tf.train.FloatList(value=[Temp[i]]))	
			#print(Temperature)
			Humidity =tf.train.Feature(float_list=tf.train.FloatList(value=[Humid[i]]))
			#The condition of normalization is done			
			mapping = {'Date_time': Date_time,
				'Temperature' : Temperature,
				'Humidity' : Humidity}
						
			example = tf.train.Example(features=tf.train.Features(feature=mapping))
			#print(example)	
			print("Normalization was executed")
			writer.write(example.SerializeToString())
			print(example)		
			
			
	else:	#For no normalization
		for i in range(len(data)):
			print("With no norm")
			Date_time = tf.train.Feature(float_list=tf.train.FloatList(value=[Date_Time[i]]))
			#print(Date_time)
			Temperature= tf.train.Feature(int64_list=tf.train.Int64List(value=[Temp[i]]))	
			#print(Temperature)
			Humidity =tf.train.Feature(int64_list=tf.train.Int64List(value=[Humid[i]]))
	
			
			#The condition of normalization is done			
			mapping = {'Date_time': Date_time,
				'Temperature' : Temperature,
				'Humidity' : Humidity}
						
			example = tf.train.Example(features=tf.train.Features(feature=mapping))
			#print(example)	
			print("Last one")
			writer.write(example.SerializeToString())
			print(example)		
			

dataset = tf.data.TFRecordDataset(filename_OUT )#buffer_size=100)
print(dataset)


# Read the data back out.

#Normalized
def decode_fn_N(example):
	print("!!!!!!!!!!!!!!!!!!!!!!!!!!!Normalized in the if")
	return tf.io.parse_single_example(
	      # Data
	      example,

	      # Schema
	      {"Date_time": tf.io.FixedLenFeature([], dtype=tf.float32),
	       "Temperature": tf.io.FixedLenFeature([], dtype=tf.float32),
	       "Humidity": tf.io.FixedLenFeature([], dtype=tf.float32)}
	  )
	  
#Not normalized	  
def   decode_fn(example):
	return tf.io.parse_single_example(
	# Data
	example,

	# Schema
	{"Date_time": tf.io.FixedLenFeature([], dtype=tf.float32),
	"Temperature": tf.io.FixedLenFeature([], dtype=tf.int64),
	"Humidity": tf.io.FixedLenFeature([], dtype=tf.int64)}
	)

if len(filename_OUT)==len("./"+args.output):
	#Not normalized

	for batch in tf.data.TFRecordDataset([filename_OUT]).map(decode_fn):
		#print(batch)
		print("Date_time = {Date_time:.2f},  Temperature = {Temperature:.2f}, Humidity = {Humidity:.2f}".format(**batch))
else: 
	
	for batch in tf.data.TFRecordDataset([filename_OUT]).map(decode_fn_N):
		#print(batch)
		print("Date_time = {Date_time:.2f},  Temperature = {Temperature:.2f}, Humidity = {Humidity:.2f}".format(**batch))
