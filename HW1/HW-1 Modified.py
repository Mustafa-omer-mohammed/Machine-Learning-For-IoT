import tensorflow as tf
import pandas as pd
import numpy as np
import time, datetime
import argparse
import os


#Calculate the normalization
Temp_MAX=50
Temp_MIN=0
Hum_Max=90
Hum_Min=20




#the input parameters for the command line
parser=argparse.ArgumentParser()

parser.add_argument('--normalize', default=False, action= 'store_true', help='normalization False/True')
parser.add_argument('--output', type=str, help='output filename')
parser.add_argument('--input', type= str, help= 'input filename')

args = parser.parse_args()

#parser info
filename=f"./{args.input}"
filename_OUT=f"./{args.output}"
normalization=args.normalize


print(f"Normalization: {normalization}")

#this is reading the filename that contains the info
data=pd.read_csv(filename, index_col=0)

#DateTime preprocessing --> create another collumn with the data and time concat
data["DateTime"]= data.Date + ' ' + data.Time 
data["DateTime"]=pd.to_datetime(data["DateTime"])

#Overwriting the information concatenated on DateTime with the values according to posix
date_time_posix=data.DateTime
data["DateTime"]=date_time_posix.apply(lambda x: time.mktime(x.timetuple())).astype("int")	#float
#print(data.DateTime[0] , type(data.DateTime[0]))

#print(data.info())
print("The correct info from the file")
print(data.head())


Temp=data.Temperature.astype("uint8")
Humid=data.Humidity.astype("uint8")
Date_Time=data.DateTime
#print(f"Date_Time {Date_Time[0]} , type {type(Date_Time[0])}" )

if normalization==True:
	
	#changing the name of the output file for the normalization
	filename_OUT= filename_OUT + "Normalized"
	print(filename_OUT)


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
		
			Date_time = tf.train.Feature(int64_list=tf.train.Int64List(value=[Date_Time[i]]))
			#print(f"Date_Time {Date_Time[i]} , type {type(Date_Time[i])}" )
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
			#print("Normalization was executed")
			writer.write(example.SerializeToString())
			#print(example)		
			
			
	else:	#For no normalization
		for i in range(len(data)):
			print("With no norm")
			Date_time = tf.train.Feature(int64_list=tf.train.Int64List(value=[Date_Time[i]]))
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
			#print("Last one")
			writer.write(example.SerializeToString())
			#print(example)		
			

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
	      {"Date_time": tf.io.FixedLenFeature([], dtype=tf.int64),
	       "Temperature": tf.io.FixedLenFeature([], dtype=tf.float32),
	       "Humidity": tf.io.FixedLenFeature([], dtype=tf.float32)}
	  )
	  
#Not normalized	  
def   decode_fn(example):
	return tf.io.parse_single_example(
	# Data
	example,

	# Schema
	{"Date_time": tf.io.FixedLenFeature([], dtype=tf.int64),
	"Temperature": tf.io.FixedLenFeature([], dtype=tf.int64),
	"Humidity": tf.io.FixedLenFeature([], dtype=tf.int64)}
	)



if len(filename_OUT)==len("./"+args.output):
	#Not normalized
	for batch in tf.data.TFRecordDataset([filename_OUT]).map(decode_fn):
		#print(batch)
		
		print("Date_time = {Date_time},  Temperature = {Temperature}, Humidity = {Humidity}".format(**batch))
						
		
else: 
	
	for batch in tf.data.TFRecordDataset([filename_OUT]).map(decode_fn_N):
		#print(batch)
		print("Date_time = {Date_time},  Temperature = {Temperature:.2f}, Humidity = {Humidity:.2f}".format(**batch))
		#print(f"Size of the {filename_OUT} is {size.st_size}")
		
size=os.stat(f'./{filename_OUT}')		
print(f"Size of the {filename_OUT} is {size.st_size} ")
	
		
