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
     
    idxs=cv2.dnn.NMSBoxes(boxes,confidences, 0.8, 0.8)
    labels_path=os.path.join("yolo", "coco.names")
    labels=open(labels_path).read().strip().split("\n")
    if(m==1):
        for i in idxs.flatten():
            (x,y)= [boxes[i][0],boxes[i][1]]
            (w,h)= [boxes[i][2],boxes[i][3]]
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
            cv2.putText(img,"{}:{}".format(labels[classIDs[i]],confidences[i]),(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,139,139),2)
    m=0  
    img=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img
'''
#cv2.imshow("Image",cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        #cv2.waitKey(0)
# Process video.
#in_clip = VideoFileClip("project_video_Trim.mp4",audio=False)
in_clip = VideoFileClip(input_path,audio=False)
out_filename = 'out.mp4'
output_path = output_path + out_filename
out_clip = in_clip.fl_image(process)
out_clip.write_videofile(output_path,audio=False)
#out_clip.write_videofile("out_filename.mp4",audio=False)
'''

	
	    

if __name__ == "__main__":
	in_clip = VideoFileClip(input_path,audio=False)
	out_clip = in_clip.fl_image(process)
	out_clip.write_videofile(output_path,audio=False)