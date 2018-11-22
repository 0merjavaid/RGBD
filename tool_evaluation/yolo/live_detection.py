from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import * 
import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

import cv2
import numpy as np 
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720) 
parser = argparse.ArgumentParser()
parser.add_argument('--image_folder', type=str, default='data/samples', help='path to dataset')
parser.add_argument('--config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
parser.add_argument('--weights_path', type=str, default='weights/yolov3.weights', help='path to weights file')
parser.add_argument('--class_path', type=str, default='data/coco.names', help='path to class label file')
parser.add_argument('--conf_thres', type=float, default=0.8, help='object confidence threshold')
parser.add_argument('--nms_thres', type=float, default=0.4, help='iou thresshold for non-maximum suppression')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_size', type=int, default=416, help='size of each image dimension')
opt = parser.parse_args()
print(opt)

os.makedirs('output', exist_ok=True)

cuda = torch.cuda.is_available()

# Set up model
model = Darknet(opt.config_path, img_size=opt.img_size)
model.load_weights(opt.weights_path)
#print (cuda)
if cuda:
    #print ("NOOOOOOOOOOOOOOO")
    model.cuda()

model.eval() # Set in evaluation mode

dataloader = DataLoader(ImageFolder(opt.image_folder, img_size=opt.img_size),
                        batch_size=opt.batch_size, shuffle=False, num_workers=opt.n_cpu)

classes = load_classes(opt.class_path) # Extracts class labels from file
print (classes)
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

imgs = []           # Stores image paths
img_detections = [] # Stores detections for each image index

print ('\nPerforming object detection:')
prev_time = time.time()


ii=0
i=0
total_time=0
total_rend_time=0
  
def run(bboxes1, bboxes2):
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou
 
while (True):
    #print  (ii,i)
    # Capture frame-by-frame
    rend0_time=time.time()
    ret, frame = cap.read()
    if not ret:
        continue
    final=frame.copy()
    final1=frame.copy()
    # print(final.shape)
    height,width,_=frame.shape
    # Our operations on the frame come here
    im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #im=frame.copy()
    im=cv2.resize(im,(416,416))
    im1 = np.pad(im, (0,0), 'constant', constant_values=127.5) / 255.
    im1 = np.transpose(im1, (2, 0, 1))
    im1 = torch.from_numpy(im1).float()
    im1 = Variable(im1.type(Tensor))
    im1=im1.view([1,3,416,416])
    with torch.no_grad():
       t0=time.time()
       detections = model(im1) 
       detections = non_max_suppression(detections, 80, opt.conf_thres, opt.nms_thres)
       t1=time.time()-t0
       total_time+=t1
       if(i%1000==0):
           print ("FPS= ",1000/total_time)
           total_time=0

       i+=1 
    hands=[]
    items=[]
    pad_x,pad_y,unpad_h,unpad_w=0,0,416,416
    if detections[0] is not None: 
       unique_labels = detections[0][:, -1].cpu().unique()
       n_cls_preds = len(unique_labels) 
       
       for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections[0]:

           box_h = ((y2 - y1) / unpad_h) * im.shape[0]
           box_w = ((x2 - x1) / unpad_w) * im.shape[1]
           y1 = ((y1 - pad_y // 2) / unpad_h) * im.shape[0]
           x1 = ((x1 - pad_x // 2) / unpad_w) * im.shape[1]
           #print (cls_pred,unique_labels,n_cls_preds)

           x1=int((float(x1)/416)*width)
           y1=int((float(y1)/416)*height)
           box_w=int((float(box_w)/416)*width)
           box_h=int((float(box_h)/416)*height)
           if(cls_pred==1):
               items.append([x1,y1,x1+box_w,y1+box_h])
               cv2.rectangle(final,(x1,y1),(x1+box_w,y1+box_h),(0,233,0),2)
           else:
               hands.append([x1,y1,x1+box_w,y1+box_h])
               cv2.rectangle(final,(x1,y1),(x1+box_w,y1+box_h),(0,0,255),2)
    if(len(hands)!=0 and len(items)!=0):

        items=np.array(items).reshape(-1,4)
        hands=np.array(hands).reshape(-1,4)
        result=run(hands,items)
        a=items[np.unravel_index(result.argmax(), result.shape)[1]]
        if(result[result!=0].shape[0]!=0):

         cv2.rectangle(final1,(a[0],a[1]),(a[2],a[3]),(0,0,255),2)

        # Display the resulting frame
        #final=cv2.resize(final,(600,450))
        #final1=cv2.resize(final1,(600,450))
        cv2.imshow('frame',final1)
    cv2.imshow('frame',final1)
    cv2.imshow('frame1',final)
    rend1_time=time.time()-rend0_time
    total_rend_time+=rend1_time

    if(ii%1000==0):
       print ("FPS rend= ", 1000/total_rend_time)
       total_rend_time=0
    ii+=1
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break


# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

