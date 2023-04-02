
# coding: utf-8

# In[2]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import numpy as np
import tensorflow as tf


# In[4]:

def values():
    tf.test.is_built_with_cuda()

    from tensorflow.python.client import device_lib
    tf.config.list_physical_devices('GPU')

    device_lib.list_local_devices()


    # In[5]:

    path= os.path.abspath(os.getcwd())
    modelw = tf.keras.models.load_model(path+'\\apnea.h5')  #change path here


    # In[6]:


    X_train = np.load(path+'\\train_input.npy', allow_pickle=True)      #change path here
    y_train = np.load(path+'\\train_label.npy', allow_pickle=True)      #change path here


    # In[7]:


    X1 = []
    X2 = []
    for index in range(len(X_train)):
        X1.append([X_train[index][0], X_train[index][1]])
        X2.append([X_train[index][2], X_train[index][3]])
    X_train1, X_train2 = np.array(X1).astype('float64'), np.array(X2).astype('float64')
    X_train1 = np.transpose(X_train1, (0, 2, 1))


    # In[8]:


    y_pred = modelw.predict([X_train1, X_train2])
    final = np.where(y_pred > 0.5, 1, 0)
    scores = modelw.evaluate([X_train1, X_train2], y_train)


    # In[ ]:


    return final
print(values())

