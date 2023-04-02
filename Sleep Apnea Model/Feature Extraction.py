
# coding: utf-8

# In[2]:

#importing the required libraries. 
#pip install wfdb
#pip install hrv==0.1.5
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import numpy as np
import csv
import wfdb
from hrv.filters import quotient, moving_median
from scipy import interpolate


# In[3]:

#constants used later
FS = 100.0
MARGIN = 10
FS_INTP = 4
MAX_HR = 300.0
MIN_HR = 20.0
MIN_RRI = 1.0 / (MAX_HR / 60.0) * 1000
MAX_RRI = 1.0 / (MIN_HR / 60.0) * 1000


# In[8]:
#Saving names of files being used
path= os.path.abspath(os.getcwd())

os.chdir(path)
data_path = '/home/era/yukkta/apnea-ecg-database-1.0.0/'   #Please change path as necessary

data_name= ['a01', 'a02', 'a03', 'a04', 'a05','a06', 'a07', 'a08', 'a09', 'a10', 'a11', 'a12', 'a13', 'a14', 'a15', 'a16', 'a17', 'a18', 'a19', 'a20', 'b01', 'b02', 'b03', 'b04', 'b05', 'c01', 'c02', 'c03', 'c04', 'c05', 'c06', 'c07', 'c08', 'c09', 'c10']

#Saving the age file
file = open("age.csv", "r")
age = list(csv.reader(file, delimiter=","))
age = age[0]
for i in range(len(age)):
    age[i] = int(age[i])
file.close()

#Saving the sex file
file = open("sex.csv", "r")
sex = list(csv.reader(file, delimiter=","))
sex = sex[0]
for i in range(len(sex)):
    sex[i] = int(sex[i])
file.close()


# In[9]:


input_array = []
label_array = []


# In[10]:

#itterating over all the datanames saved earlier.
for data_index in range(len(data_name)):
    win_num = len(wfdb.rdann(os.path.join(data_path,data_name[data_index]), 'apn').symbol) #reads the apnea annotations
    signals, fields = wfdb.rdsamp(os.path.join(data_path,data_name[data_index])) #extracts the signals and their field names from the fed data.
    for index in range(1, win_num):
        samp_from = index * 60 * FS # 60 seconds duration.         
        samp_to = samp_from + 60 * FS  # as we need 1 min worth of data.
        # every min of data with a margin of 10 secs on both ends
        qrs_ann = wfdb.rdann(data_path + data_name[data_index], 'qrs', sampfrom=samp_from - (MARGIN*100), sampto=samp_to + (MARGIN*100)).sample
        apn_ann = wfdb.rdann(data_path + data_name[data_index], 'apn', sampfrom=samp_from, sampto=samp_to-1).symbol
        
        # extracting the peaks (mmaximum values) during the sampling interval.
        interval = int(FS * 0.250)
        qrs_amp = []
        for index in range(len(qrs_ann)):
            curr_qrs = qrs_ann[index]
            amp = np.max(signals[curr_qrs-interval:curr_qrs+interval])
            qrs_amp.append(amp)
            
    rri = np.diff(qrs_ann)
    rri_ms = rri.astype('float') / FS * 1000.0
        
    rri_filt = moving_median(rri_ms)
    
    if len(rri_filt) > 5 and (np.min(rri_filt) >= MIN_RRI and np.max(rri_filt) <= MAX_RRI):

        rri_time = np.cumsum(rri_filt) / 1000.0  # make it seconds
        time_rri = rri_time - rri_time[0]  
        
        #Digitizing the data
        time_rri_interp = np.arange(0, time_rri[-1], 1 / float(FS_INTP))
        tck = interpolate.splrep(time_rri, rri_filt, s=0)
        rri_interp = interpolate.splev(time_rri_interp, tck, der=0)
        time_intp, rri_intp = time_rri_interp, rri_interp

        #interpolating the data using the known data points
        time_qrs = qrs_ann / float(FS)
        time_qrs = time_qrs - time_qrs[0]
        time_qrs_interp = np.arange(0, time_qrs[-1], 1/float(FS_INTP))
        tck = interpolate.splrep(time_qrs, qrs_amp, s=0)
        qrs_interp = interpolate.splev(time_qrs_interp, tck, der=0)
        qrs_time_intp, qrs_intp = time_qrs_interp, qrs_interp


        rri_intp = rri_intp[(time_intp >= MARGIN) & (time_intp < (60+MARGIN))]
        qrs_intp = qrs_intp[(qrs_time_intp >= MARGIN) & (qrs_time_intp < (60 + MARGIN))]

#Forming the labels for the training.
        if len(rri_intp) != (FS_INTP * 60):
            skip = 1
        else:
            skip = 0

        if skip == 0:
            rri_intp = rri_intp - np.mean(rri_intp)
            qrs_intp = qrs_intp - np.mean(qrs_intp)
            if apn_ann[0] == 'N': # Normal
                label = 0.0
            elif apn_ann[0] == 'A': # Apnea
                label = 1.0
            else:
                label = 2.0
                
#merging the waveform related data as well as age and sex together into one single data file
            input_array.append([rri_intp, qrs_intp, age[data_index], sex[data_index]])
            label_array.append(label)
np.save('input.npy', input_array)
np.save('label.npy', label_array)

