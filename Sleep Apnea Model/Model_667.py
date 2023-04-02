
# coding: utf-8

# In[2]:



# In[3]:

#importing necessary libraries.
#pip install wfdb
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import numpy as np
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import wfdb
from sklearn.utils import class_weight


# In[4]:

#to check if code is running or GPU
tf.test.is_built_with_cuda()

from tensorflow.python.client import device_lib
tf.config.list_physical_devices('GPU')

device_lib.list_local_devices()


# In[5]:

# To store weights of epoch with best accuracy.
save_here = os.path.join("/home/era/yukkta", "apn.h5")
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=save_here,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


# In[6]:

# Loading and reshaping data to fit the model 
sequence_length = 240

X_train = np.load('/home/era/yukkta/train_input.npy', allow_pickle=True)  #change path as necessary.
y_train = np.load('/home/era/yukkta/train_label.npy', allow_pickle=True)

X_test = np.load('/home/era/yukkta/test_input.npy', allow_pickle=True)
y_test = np.load('/home/era/yukkta/test_label.npy', allow_pickle=True)

# The waveform related data is stored in X1
# Sex and age related data is stored in X2.
X1 = []
X2 = []
for index in range(len(X_train)):
    X1.append([X_train[index][0], X_train[index][1]])
    X2.append([X_train[index][2], X_train[index][3]])
X_train1, X_train2 = np.array(X1).astype('float64'), np.array(X2).astype('float64')



X1 = []
X2 = []
for index in range(len(X_test)):
    X1.append([X_test[index][0], X_test[index][1]])
    X2.append([X_test[index][2], X_test[index][3]])
X_test1, X_test2 = np.array(X1).astype('float64'), np.array(X2).astype('float64')

#data is reshaped to fit the model input dimensions
X_train1 = np.transpose(X_train1, (0, 2, 1))

X_test1 = np.transpose(X_test1, (0, 2, 1))

'''since most patients do not have sleep apnea, it is ensential to ensure that the model is not 
biased towards the more probable prediction. Therefore class weights are used.'''
class_w = class_weight.compute_class_weight(class_weight = "balanced", classes = np.unique(y_train), y = y_train)


# In[7]:

#MODEL ARCHETECTURE
layers = {'input': 2, 'hidden1': 256, 'hidden2': 256, 'output': 1}
x1 = tf.keras.layers.Input(shape=(sequence_length, layers['input']))
m1 = tf.keras.layers.LSTM(layers['hidden1'],                   
                recurrent_dropout=0.5,
               return_sequences=True)(x1)

m1 = tf.keras.layers.LSTM(
        layers['hidden2'],
        recurrent_dropout=0.5,
        return_sequences=False)(m1)

x2 = tf.keras.layers.Input(shape=(2,))
m2 = tf.keras.layers.Dense(32)(x2)

merged = tf.keras.layers.Concatenate(axis=1)([m1, m2])

out = tf.keras.layers.Dense(8)(merged)
out = tf.keras.layers.Dense(layers['output'], kernel_initializer='normal')(out)
out = tf.keras.layers.Activation("sigmoid")(out)


model = tf.keras.models.Model(inputs=[x1, x2], outputs=[out])

#loss function and optimizer
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics = ['accuracy'])


model.summary()


# In[8]:

#Training starts
class_w = {i : class_w[i] for i in range(2)}
history = model.fit([X_train1, X_train2], y_train, epochs=20, batch_size=256, validation_split=0.1, class_weight=class_w, callbacks=[cp_callback])

