# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:00:00 2018
@author: Muhammad Umer Javaid (omer.javaid@visionx.io)
"""

from __future__ import division
from yolo.models import *
import torch
from torch.autograd import Variable
import cv2
import numpy as np
from yolo.utils import *


class YoloInference:

    def __init__(self, model_path="yolo/final.weight", conf_thres=0.35, batch_size=1, yolo_cfg="yolo/smartcart.cfg",
                 nms_thres=0.25, img_size=416):
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres
        # Extracts class labels from file

        self.model = Darknet(yolo_cfg, img_size=img_size)
        self.model.load_weights(model_path)
        if torch.cuda.is_available():
            self.tensor = torch.cuda.FloatTensor
            self.model.cuda()
        else:
            self.tensor = torch.FloatTensor
        self.model.eval()

    def yolo_localize(self, images):
        height, width, _ = images[0].shape
        images_list = list()
        for image in images:
            im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image.shape
            im = cv2.resize(im, (416, 416))
            img_for_model = np.pad(
                im, (0, 0), 'constant', constant_values=127.5) / 255.
            img_for_model = np.transpose(img_for_model, (2, 0, 1))
            images_list.append(img_for_model)
        img_for_model = np.array(images_list).reshape(-1, 3, 416, 416)
        img_for_model = torch.from_numpy(img_for_model).float()
        img_for_model = Variable(img_for_model.type(self.tensor))
        img_for_model = img_for_model.view([len(images), 3, 416, 416])

        with torch.no_grad():
            detections = self.model(img_for_model)
            detections = non_max_suppression(
                detections, 80, self.conf_thres, self.nms_thres)
        pad_x, pad_y, unpad_h, unpad_w = 0, 0, 416, 416

        batch_detections = list()
        for detection in detections:
            boxes = list()
            if detection is not None:
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
                    box_h = ((y2 - y1) / unpad_h) * 416
                    box_w = ((x2 - x1) / unpad_w) * 416
                    y1 = ((y1 - pad_y // 2) / unpad_h) * 416
                    x1 = ((x1 - pad_x // 2) / unpad_w) * 416
                    x1 = int((float(x1) / 416) * width)
                    y1 = int((float(y1) / 416) * height)
                    box_w = int((float(box_w) / 416) * width)
                    box_h = int((float(box_h) / 416) * height)
                    label = cls_pred.item()
                    obj = np.array([x1, y1, x1 + box_w, y1 + box_h])
                    obj[obj < 0] = 0
                    obj = np.hstack((label, obj))
                    boxes.append(obj.astype("uint16"))
                current_boxes = np.array(boxes).astype("uint16")
                assert (current_boxes >= 0).all()
                batch_detections.append(current_boxes)
            else:
                batch_detections.append(
                    np.array([0, 0, 0, 0, 0]).astype("uint16"))
        return batch_detections
