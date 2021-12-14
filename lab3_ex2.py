import argparse
import os
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True, help='model name')
parser.add_argument('--mfcc', action='store_true', help='use MFCCs')
parser.add_argument('--silence', action='store_true', help='add silence')
args = parser.parse_args()

# {'mlp', 'cnn', 'ds_cnn'}
# python lab3_ex2.py --model mlp --mfcc 

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

# Exercise 2.1
if args.silence is True:
    data_dir = os.path.join('.', 'data', 'mini_speech_commands_silence')
else:
    zip_path = tf.keras.utils.get_file(
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        fname='mini_speech_commands.zip',
        extract=True,
        cache_dir='.', cache_subdir='data')

    data_dir = os.path.join('.', 'data', 'mini_speech_commands')

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)

if args.silence is True:
    total = 9000
else:
    total = 8000

# Exercise 2.2
train_files = filenames[:int(total*0.8)]
val_files = filenames[int(total*0.8): int(total*0.9)]
test_files = filenames[int(total*0.9):]

LABELS = np.array(tf.io.gfile.listdir(str(data_dir)))                                                   
LABELS = LABELS[LABELS != 'README.md']

class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate                                             # 16000  
        self.frame_length = frame_length                                               # 640 
        self.frame_step = frame_step                                                   # 320 
        self.num_mel_bins = num_mel_bins                                               # 40 
        self.lower_frequency = lower_frequency                                         # 20 
        self.upper_frequency = upper_frequency                                         # 4000
        self.num_coefficients = num_coefficients                                       # 10 
        num_spectrogram_bins = (frame_length) // 2 + 1                                  # ( frame size // 2 ) + 1 

        '''
        STFT_OPTIONS = {'frame_length': 256, 'frame_step': 128, 'mfcc': False}
        MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}
        '''

        if mfcc is True:                                                                # Remember we need to compute this matrix once so it will be a class argument 
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                    self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]                                  # -1 is audio.wav so 
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        return audio, label_id

    def pad(self, audio):
        # Padding for files with less than 16000 samples
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)     # if the shape of the audio is already = 16000 (sampling rate) we will add nothing 

        # Concatenate audio with padding so that all audio clips will be of the  same length
        audio = tf.concat([audio, zero_padding], 0)
        # Unify the shape to the sampling frequency (16000 , )
        audio.set_shape([self.sampling_rate])

        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
 
   # expand_dims will not add or reduce elements in a tensor, it just changes the shape by adding 1 to dimensions. For example, a vector with 10 elements could be treated as a 10x1 matrix.
    #The situation I have met to use expand_dims is when I tried to build a ConvNet to classify grayscale images. The grayscale images will be loaded as matrix of size [320, 320]. However,
    #tf.nn.conv2d require input to be [batch, in_height, in_width, in_channels], 
    #where the in_channels dimension is missing in my data which in this case should be 1. So I used expand_dims to add one more dimension.
    
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls = 4) # better than 4 tf.data.experimental.AUTOTUNE
        ds = ds.batch(32)
        ds = ds.cache()
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds

STFT_OPTIONS = {'frame_length': 256, 'frame_step': 128, 'mfcc': False}
MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}
if args.mfcc is True:
    options = MFCC_OPTIONS
    strides = [2, 1]
else:
    options = STFT_OPTIONS
    strides = [2, 2]

generator = SignalGenerator(LABELS, 16000, **options)
train_ds = generator.make_dataset(train_files, True)
val_ds = generator.make_dataset(val_files, False)
test_ds = generator.make_dataset(test_files, False)


if args.silence is True:
    units = 9
else:
    units = 8

# Exercise 2.3
mlp = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units = 256, activation='relu'),
    tf.keras.layers.Dense(units = 256, activation='relu'),
    tf.keras.layers.Dense(units = 256, activation='relu'),
    tf.keras.layers.Dense(units = units)
])

cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=128, kernel_size=[3,3], strides=strides, use_bias=False),
    tf.keras.layers.BatchNormalization(momentum=0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters=128, kernel_size=[3,3], strides=[1,1], use_bias=False),
    tf.keras.layers.BatchNormalization(momentum=0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.Conv2D(filters=128, kernel_size=[3,3], strides=[1,1], use_bias=False),
    tf.keras.layers.BatchNormalization(momentum=0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units = units)
])

ds_cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=256, kernel_size=[3,3], strides=strides, use_bias=False),
    tf.keras.layers.BatchNormalization(momentum=0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
    tf.keras.layers.Conv2D(filters=256, kernel_size=[1,1], strides=[1,1], use_bias=False),
    tf.keras.layers.BatchNormalization(momentum=0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
    tf.keras.layers.Conv2D(filters=256, kernel_size=[1,1], strides=[1,1], use_bias=False),
    tf.keras.layers.BatchNormalization(momentum=0.1),
    tf.keras.layers.ReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(units = units)
])

# Exercise 2.4
MODELS = {'mlp': mlp, 'cnn': cnn, 'ds_cnn': ds_cnn}
model = MODELS[args.model]

tb_run = 0 

loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam()
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
if args.mfcc is False:
    checkpoint_filepath = './checkpoints/stft/chkp_best_'+args.model
    tb_log_1 = f'logs/tb_log_STFT_{args.model}'
else:
    checkpoint_filepath = './checkpoints/mfcc/chkp_best_' + args.model
    tb_log_1 = f'logs/tb_log__MFCC{args.model}'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_sparse_categorical_accuracy',
    verbose=1,
    mode='max',
    save_best_only=True,
    save_freq='epoch'
)
tb_callback = keras.callbacks.TensorBoard(log_dir= tb_log_1 , histogram_freq=1 , profile_batch = '200,220')
model.fit(train_ds, validation_data=val_ds, epochs=5,  callbacks=[tb_callback ,model_checkpoint_callback ])

tb_run += 1



model.summary()

loss, error = model.evaluate(test_ds)
print('Error: ', error)

best_model = tf.keras.models.load_model(checkpoint_filepath)
best_model.evaluate(test_ds)

# if not os.path.exists('./models'):
#     os.makedirs('./models')
#
# run_model = tf.function(lambda x: model(x))
# concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2], tf.float32))
# saving_path = os.path.join('.','models', '{}_{}'.format(LABELS, args.model))
# model.save(saving_path, signatures=concrete_func)