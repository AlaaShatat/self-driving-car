#### import some libraries
import sys
import numpy as np
import time
import cv2
import os
import glob
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

# input and output path
input_path = sys.argv[1]
output_path = sys.argv[2]



weights_path = os.path.join("yolo", "yolov3.weights")
config_path = os.path.join("yolo", "yolov3.cfg")

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
names = net.getLayerNames()


def process(img):
    #image_path = os.path.join("yolo", "test_img.png")
    #img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    (H, W) = img.shape[:2]
    layers_names = [names[i - 1] for i in net.getUnconnectedOutLayers()]
    blob=cv2.dnn.blobFromImage(img,1/255.0, (416,416), crop=False, swapRB=False)
    net.setInput(blob)
    ##calculate the runtime of the algorithm 
    start_t=time.time()
    layers_output=net.forward(layers_names)
    
    boxes = []
    confidences = []
    classIDs = []
    m=0  
    for output in layers_output:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if(confidence > 0.85):
                m=1 
                box = detection[:4] * np.array([W, H, W, H])
                bx, by, bw, bh = box.astype("int")
                x = int(bx-(bw / 2))
                y = int(by - (bh / 2))
                boxes.append([x, y, int(bw), int(bh)])
                confidences.append(float(confidence))
                classIDs.append(classID)
               