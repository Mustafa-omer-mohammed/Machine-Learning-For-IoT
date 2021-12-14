
from tensorflow import keras
import os
import numpy as np
import os
import tensorflow as tf

class CreateModel : 

    '''  
    - We need 3 models so model name should be as input  
    - we dont need the input shapes in our case it's (6,2) matric because flatten will output (,12)
        and in case of conv layers it works in any shape sice it depends on filters  
    - we need the output shape which depends on the labels argument of the in if  
        1- labels == 2 output layer should be set to 2 
        2- otherwise output layer set to 1  '''
 
    def __init__(self   , labels, train_ds,  val_ds, batch_train, test_ds,epoch ) :
        
        self.labels = labels
        # self.target_train=target_train
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.batch_train = batch_train
        self.epoch = epoch
        self.test_ds = test_ds

    # first Model MLP 
    def MLP (self) :
        if self.labels < 2 :
            output_units = 1
            metruc
        else :
            output_units = 2
        model = keras.Sequential([
                keras.layers.Flatten(), # do we need the input shapes () ?   
                keras.layers.Dense(128, activation='relu', name='first_dense'),
                keras.layers.Dense(128, activation='relu', name='Second_dense'),
                keras.layers.Dense(units = output_units , name='Output_dense')
                ])

        model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])
        history = model.fit(self.train_ds,  epochs=self.epoch, validation_data=self.val_ds, batch_size=len(self.batch_train))
        results = model.evaluate(self.test_ds)

        return model,history

    def CNN_1D(self) :
        if self.labels < 2 :
            output_units = 1
            metric = 
        else :
            output_units = 2

        model = keras.Sequential([
            keras.layers.Conv1D(filters = 64 , kernel_size = 3 , activation='relu' , name = 'Conv1D' ),
            keras.layers.Flatten(),
            keras.layers.Dense(128, activation='relu', name='first_dense'),
            keras.layers.Dense(units = output_units ,  name='Output_dense')
                                    ])   
        model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])
        history = model.fit(self.train_ds,   epochs=self.epoch, validation_data=self.val_ds, batch_size=len(self.batch_train))
        results = model.evaluate(self.test_ds)

        return model,history
                                    
    def LSTM (self) :
        if self.labels < 2 :
            output_units = 1
        else :
            output_units = 2
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units=64, name='LSTM_Layer'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units = output_units ,  name='Output_dense')
                                    ])

        model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mae'])
        history = model.fit(self.train_ds,    epochs=self.epoch, validation_data=self.val_ds, batch_size=len(self.batch_train))
        results = model.evaluate(self.test_ds)

        return model,history

        