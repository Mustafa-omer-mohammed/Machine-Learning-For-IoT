import tensorflow as tf 
from tensorflow import keras
import argparse
import numpy as np
import os
import pandas as pd

#importing models
from my_modles_Class import CreateModel
from  lab3_ex1_draft import WindowGenerator


def main() :

    print("START OF THE PROGRAM")

    '''parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='model name')
    parser.add_argument('--labels', type=int, required=True, help='model output')

    args = parser.parse_args()

    model = args.model
    lables = args.labels'''

    zip_path = tf.keras.utils.get_file(
    origin=None,
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')


    csv_path, _ = os.path.splitext(zip_path) # this how you remove the .zip at the end of the name 
    df = pd.read_csv(csv_path)

    column_indices = [2, 5]                           # temperature and humidity column_indices
    columns = df.columns[column_indices]              # temperature and humidity column_Names
    data = df[columns].values.astype(np.float32)      # extract only numpy arrays

    # Train ,Validation ,Test Split  we don't have to shuffle because we don't have the labels yet: 
    n = len(data)                          
    train_data = data[0:int(n*0.7)]
    val_data = data[int(n*0.7):int(n*0.9)]
    test_data = data[int(n*0.9):]

    mean = train_data.mean(axis=0)       # remember these used for normalization but only calculated form training data   
    std = train_data.std(axis=0)

    input_width = 6
    # LABEL_OPTIONS = args.labels

    
    # Going around the labels
    labels = [0,1,2]
    Results = {} 

    
    for label in  labels:

        if label == 0 :
            print("*" * 20,"Computing for Temp","*" * 20)

        if label == 1 :
            print("*" * 20,"Computing for Hum" , "*" * 20)

        generator = WindowGenerator(input_width, label, mean, std)
        train_ds = generator.make_dataset(train_data, True)
        val_ds = generator.make_dataset(val_data, False)
        test_ds = generator.make_dataset(test_data, False)
        batch_train=next(iter(train_ds))

        batch_test=next(iter(test_ds))
        inp_train=batch_train[0]
        inp_test=batch_test[0]
        inp_test
        target_train=batch_train[1]
        target_test=batch_test[1]
        target_test

        #print(f"the training dataset : {train_ds}")

        model=CreateModel(label, train_ds, val_ds, batch_train,test_ds, epoch=2)

        MLP, MLP_history = model.MLP()
        print("*" * 25 , "exEcuting MLB :")
        CNN_1D, CNN_1D_history = model.CNN_1D()
        print("*" * 25 , " exEcuting CNN_1D  :")
        LSTM, LSTM_history = model.LSTM()
        print("*" * 25 , "exEcuting LSTM :")

        MLP_SUMMARY  = MLP.summary()
        CNN_1D_SUMMARY = CNN_1D.summary()
        LSTM_SUMMARY   = LSTM.summary()
        if label == 0:
            #"MLP_TEMP" , "CNN_1D_TEMP" , "LSTM_TEMP"
            print(f"MLP_history:{MLP_SUMMARY}")       
        
        elif label == 1:
            #"MLP_HUM" , "CNN_1D_HUM" , "LSTM_HUM" 

            print(f"CNN_1D_history: {CNN_1D}")   
        else: 
            # "MLP_TEMP_HUM" , "CNN_1D_TEMP_HUM" , "LSTM_TEMP_HUM"
            print(f"LSTM_history: {LSTM_SUMMARY}")
            print("TST")   


if __name__ == "__main__" :
    main()