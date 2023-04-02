import http.server
import os
import logging
import http.server as server
import socket
 
import socketserver
 
import webbrowser
 
import pyqrcode
from pyqrcode import QRCode
import json
import statistics
import Prediction
import snore

import png
# assigning the appropriate port value
PORT = 8000
# this finds the name of the computer user

desktop = os.path.abspath(os.getcwd())
os.chdir(desktop)

hostname = socket.gethostname()
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(("8.8.8.8", 80))
IP = "http://" + s.getsockname()[0] + ":" + str(PORT)

link = IP
url = pyqrcode.create(link)
url.svg("myqr.svg", scale=8)

#extracting output from models
res=Prediction.values()
result=[i for i in res]
sno=snore.value()
snore=[i for i in sno]
# result=[1,0,1,0,1,1,1,1,0]


class HTTPRequestHandler(server.SimpleHTTPRequestHandler):

    def do_OPTIONS(self):
        # print(self.headers)
        # self.send_response(204, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET,POST,PUT,OPTIONS')
        # self.send_header("Access-Control-Allow-Headers", "X-Requested-With")
        self.send_header("Access-Control-Allow-Headers", "Authorization")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        # self.send_header("Access-Control-Allow-Headers", "Options")
        self.end_headers()
        
        data=[]
        with open(self.path[2:],"r") as f:
            lines=f.readlines()
            for line in lines:
                data.append(int(line.rstrip()))

        self.wfile.write(bytes(json.dumps(data),'utf-8'))

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin','*')
        server.SimpleHTTPRequestHandler.end_headers(self)

    def do_GET(self):
        # return the content of the file requested

        # print(self)
        self.send_response(200)
        self.send_header('Content-type','text/html')
        self.end_headers()
        datajson=""
        data=[]

        
        if self.path[2:]=="ecg001.csv":
            with open(self.path[2:],"r") as f:
                lines=f.readlines()
                for line in lines:
                    data.append(line.rstrip())
            self.wfile.write(bytes(json.dumps(data),'utf-8'))

        elif self.path[2:]=="user001.json":
            with open(self.path[2:],"r") as f: 
                lines=f.readlines()
                for line in lines:
                    datajson+=(line.rstrip().lstrip())
            self.wfile.write(bytes(json.dumps(datajson),'utf-8'))
            print


    def do_PUT(self):
        """Save a file following a HTTP PUT request"""
        filename = os.path.basename(self.path)


        file_length = int(self.headers['Content-Length'])
        data=[]

        with open(filename,'r') as f:
            data=f.readlines()
            if (len(data)>=100):
                data=data[20:]
        
        with open(filename,"w") as f:
            for i in data:
                f.write(i)
        with open(filename, 'ab') as output_file:
            output_file.write(self.rfile.read(file_length))

        self.send_response(201, 'Created')
        self.end_headers()
        reply_body = 'Saved "%s"\n' % filename
        self.wfile.write(reply_body.encode('utf-8'))

        #update the user json file storing final info

        dict=json.load(open("user001.json","r"))
        dict["apnea"]["presence"]=0
        dict["apnea"]["events"]=0

        if (self.path[2:]=="pre001.csv"):
            for i in range(len(data)):
                data[i]=int(data[i].rstrip())
            var=10000
            for i in range(10,len(data)):
                if (statistics.variance(data[i-10:i])<var):
                    var=statistics.variance(data[i-10:i])
                    dict["Deep-sleep"]["start"]=i-10
                    dict["Deep-sleep"]["end"]=i
        

        elif (self.path[2:]=="ecg001.csv"):
            count_events=0
            for i in range(len(result)):
                if result[i]==1:
                    count_events+=1
            if (count_events):
                dict["apnea"]["presence"]=1
                dict["apnea"]["events"]=count_events
            count_snore=0
            for i in range(len(snore)):
                if snore[i]>0.5:
                    count_snore+=1
                if (count_snore):
                    dict["snore"]=count_snore

        # write the jsons
        json.dump(dict,open("user001.json","w"))


with socketserver.TCPServer(("", PORT), HTTPRequestHandler) as httpd:
    print("serving at port", PORT)
    print("Type this in your Browser", IP)
    print("or Use the QRCode")
    httpd.serve_forever()