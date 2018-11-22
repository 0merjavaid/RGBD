from __future__ import division

from yolo.models import *
from yolo.utils.utils import *
from yolo.utils.datasets import * 
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

print ("imported")
class YOLO:
    def __init__(self,model_path="yolo/checkpoints/final.weight"):
        self.cuda = torch.cuda.is_available()
        self.Tensor = torch.cuda.FloatTensor if self.cuda else torch.FloatTensor
        self.model= self.get_model(model_path)
        self.classes = load_classes("yolo/data/coco.names")
        

    def get_model(self,model_path):
        if not os.path.exists(model_path):
            print ("YOlO path Not found")
            return False

        # Set up model
        model = Darknet("yolo/config/yolov3.cfg", img_size=(416,416))
        model.load_weights(model_path)
        #print (cuda)
        if self.cuda: 
            model.cuda()
        

        model.eval() # Set in evaluation mode
        return model
        
    def get_item_of_interest(self,frame=None):


  

        imgs = []           # Stores image paths
        img_detections = [] # Stores detections for each image index
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
        im1 = Variable(im1.type(self.Tensor))
        im1=im1.view([1,3,416,416])
        with torch.no_grad():
           t0=time.time()
           detections = self.model(im1) 
           detections = non_max_suppression(detections, 80, 0.5, 0.45)
          
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
                   cv2.rectangle(final1,(x1,y1),(x1+box_w,y1+box_h),(0,233,0),2)
               else:
                   hands.append([x1,y1,x1+box_w,y1+box_h])
                   cv2.rectangle(final1,(x1,y1),(x1+box_w,y1+box_h),(0,0,255),2)
        # if(len(hands)!=0 and len(items)!=0):
                        
        #     items=np.array(items).reshape(-1,4)
        #     hands=np.array(hands).reshape(-1,4)
        #     result=self.run(hands,items)
        #     a = hands[np.unravel_index(result.argmax(), result.shape)[0]]
        #     if(result[result!=0].shape[0]!=0):

        #         cv2.rectangle(final1,(a[0],a[1]),(a[2],a[3]),(0,0,255),2)
        # print (len(hands))
        #return a, np.array(items).reshape(-1,4)
        return np.array(hands).reshape(-1,4), np.array(items).reshape(-1,4)
        #return None
 
 
    def run(self,bboxes1, bboxes2):
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
 
 

 