import os
from matplotlib import pyplot as plt
import tensorflow as tf 
import tensorflow_io as tfio
import math
from IPython.display import Audio
from string import ascii_uppercase
from pandas import DataFrame
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.io.wavfile import write
import librosa.display
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten,MaxPooling2D,Dropout,GlobalAveragePooling2D,Activation
import math
#import visualkeras
#Definging the path to the dataset

DATA_SNORING_PATH = os.path.join('snoring','1')
NOT_DATA_SNORING_PATH = os.path.join('snoring','0')

#defining the path to the single audio file

SNORING_FILE = os.path.join(DATA_SNORING_PATH,'1_0.wav')
NOT_SNORING_FILE = os.path.join(NOT_DATA_SNORING_PATH,'0_0.wav')


#function to resample the audio file

def load_wav_16k_mono(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels) 
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

wave = load_wav_16k_mono(SNORING_FILE)
nwave = load_wav_16k_mono(NOT_SNORING_FILE)

## plot 44100Hz wave to time


# plt.rcParams["figure.figsize"] = [7.50, 3.50]
# plt.rcParams["figure.autolayout"] = True
# input_data = read(SNORING_FILE  )
# x_1 = np.linspace(0, 1, 44100)

# audio = input_data[1]
# plt.plot(x_1,audio[0:44100])
# plt.ylabel("Amplitude")
# plt.xlabel

## plot 16000Hz wave to time


# x = np.linspace(0, 1, 16000)
# plt.figure(figsize=(14, 6))
# plt.plot(x,wave, alpha=0.7)
# plt.plot(x,nwave, alpha=0.7)
# plt.xlabel('Time(Sec)')
# plt.ylabel('Amplitude')
# plt.legend(labels=['Snoring Wave', 'Not Snoring Wave'])
# plt.xticks(np.linspace(0, 1, 11))
# plt.show()

pos = tf.data.Dataset.list_files(DATA_SNORING_PATH+'/*.wav')
neg = tf.data.Dataset.list_files(NOT_DATA_SNORING_PATH+'/*.wav')

positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
data = positives.concatenate(negatives)

#exploring the dataset wave length

lengths = []
for file in os.listdir(os.path.join(NOT_DATA_SNORING_PATH)):
    tensor_wave = load_wav_16k_mono(os.path.join(NOT_DATA_SNORING_PATH, file))
    lengths.append(len(tensor_wave))


tf.math.reduce_mean(lengths)
tf.math.reduce_min(lengths)
tf.math.reduce_max(lengths)
#function to convert the audio file to spectrogram

def preprocess(file_path, label): 
    wav = load_wav_16k_mono(file_path)
    wav = wav[:16000]
    zero_padding = tf.zeros([16000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

filepath, label = negatives.shuffle(buffer_size=10000).as_numpy_iterator().next()
#positives = Snoring , negatives = Not Snoring
wav = load_wav_16k_mono(filepath)
wav = wav[:16000]
wav
spectrogram, label = preprocess(filepath, label)

#plotting the spectrogram

# plt.figure(figsize=(15,8))
# plt.imshow(tf.transpose(spectrogram)[0])
# plt.gca().invert_yaxis()
# plt.show()

data.as_numpy_iterator().next()

#preprocessing the dataset and splitting it into batches

data = data.map(preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(64)
data = data.prefetch(8)

#adjusting the shape of the dataset

train = data.take(math.ceil(len(data)*.7))
test = data.skip(math.ceil(len(data)*.7)).take(math.floor(len(data)*.3))

samples, labels = train.as_numpy_iterator().next()

input_shape = samples.shape[1:]



#building the model
def value():
	model = Sequential()
	model.add(Conv2D(16, (3,3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(16, (3,3), activation='relu'))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	#compiling the model

	model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

	hist = model.fit(train, epochs=5, validation_data=test)

	return model.predict(test)
