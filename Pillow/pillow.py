import requests
import os

url="http://172.17.76.7:8000//"
file_local_ecg="data_ecg.csv"
file_local_pre="data_pre.csv"

user="001"
file_server=user+".csv"

# store initial time stamp of modification
stat_ecg=os.stat(file_local_ecg).st_mtime
stat_presure=os.stat(file_local_pre).st_mtime


while 1:
	if (stat_ecg!=os.stat(file_local_ecg).st_mtime): # check if modified time stamp changes
		r = requests.put(url+"ecg"+file_server, data=open(file_local_ecg, 'rb'))
		stat_ecg=os.stat(file_local_ecg).st_mtime
		print("sent_ecg")
		
	if (stat_presure!=os.stat(file_local_pre).st_mtime):
		r = requests.put(url+"pre"+file_server, data=open(file_local_pre, 'rb'))
		stat_presure=os.stat(file_local_pre).st_mtime
		print("sent_pre")