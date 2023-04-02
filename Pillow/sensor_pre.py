import time,random

numr=20
data=[]
index=0
ref=[i for i in range(12)]+[i for i in range(12,0,-1)]

def reading_from_sensor():
	global index
	time.sleep(0.2)
	index+=1
	# return ref[index%(len(ref))]
	return random.randint(0,15)


file="data_pre.csv"
open(file,"w").close()
flag=1
while flag:
	#genereate data
	reading=reading_from_sensor()
	if (len(data)>=numr):
		with open(file,"w") as f:
			for i in data:
				f.write(str(i)+"\n")
		data=[]
	data.append(reading)