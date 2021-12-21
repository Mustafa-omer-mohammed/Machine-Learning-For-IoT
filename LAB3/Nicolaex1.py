import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from tensorflow import keras


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='model name')
parser.add_argument('--labels', type=int, required=True, help='model output')
args = parser.parse_args()

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)

column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

input_width = 6
LABELS = args.labels

class WindowGenerator:
    def __init__(self, input_width, label_options, mean, std):
        self.input_width = input_width
        self.label_options = label_options
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features):
        inputs = features[:, :-1, :]

        if self.label_options < 2:
            labels = features[:, -1, self.label_options]
            labels = tf.expand_dims(labels, -1)
            num_labels = 1
        else:
            labels = features[:, -1, :]
            num_labels = 2

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, num_labels])

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=input_width+1,
                sequence_stride=1,
                batch_size=32)
        ds = ds.map(self.preprocess)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds

# Exercise 1.7
class MultiOutputMAE(tf.keras.metrics.Metric):
    def __init__(self, name='mean_absolute_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros', shape=(2,))
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_pred - y_true)
        error = tf.reduce_mean(error, axis=0)
        self.total.assign_add(error)
        self.count.assign_add(1.)

        return

    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))

    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)

        return result

generator = WindowGenerator(input_width, LABELS, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)

if LABELS < 2:
    units = 1
else:
    units=2

# Exercise 1.3
mlp = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(units = units)
])

cnn = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=units)
])

lstm = tf.keras.Sequential([
    tf.keras.layers.LSTM(units=64),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=units)
])

# Exercise 1.4
MODELS = {'mlp': mlp, 'cnn': cnn, 'lstm': lstm}
model = MODELS[args.model]

loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam()

if LABELS < 2:
    metrics = [tf.keras.metrics.MeanAbsoluteError()]
else:
    metrics = [MultiOutputMAE()]
model.compile(loss = loss, optimizer = optimizer, metrics = metrics)


tb_run = 0
tb_callback = keras.callbacks.TensorBoard(log_dir='./tb_log/run_{}'.format(tb_run), histogram_freq=1)
#history = model.fit(inp_train,target_train,  epochs=5,validation_data=val_ds ,batch_size=len(batch_train), callbacks=[tb_callback])
model.fit(train_ds, epochs=20,   validation_data=val_ds,callbacks=[tb_callback])

tb_run += 1

print(model.summary())

test_loss, test_error = model.evaluate(test_ds)
print('Test error: ', test_error)

# Save the model on disk
if not os.path.exists('./models'):
    os.makedirs('./models')

run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2], tf.float32))
saving_path = os.path.join('.','models', '{}_{}'.format(LABELS, args.model))
model.save(saving_path, signatures=concrete_func)

if LABELS == 2:

    converter = tf.lite.TFLiteConverter.from_saved_model(saving_path)
    tflite_model = converter.convert()

    tflite_model_dir = os.path.join('.','models', '{}_{}.tflite'.format(LABELS, args.model))

    with open(tflite_model_dir, 'wb') as fp:
        fp.write(tflite_model)
