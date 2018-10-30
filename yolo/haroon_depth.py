import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import yolo for object detection
import yolo
from operator import itemgetter

from depth_utils import *

# Setup depth stream
pipeline, config = setup_pipeline()
align = align_frames(pipeline, config)
color_frame, aligned_depth_frame = get_aligned_frames(pipeline, align)
color_image, depth_image = rgbd_to_numpy(color_frame, aligned_depth_frame)

# Init YOLO model
YOLO = yolo.YOLO()

try:
    while True:
        # Get frameset of color and depth
        color_frame, aligned_depth_frame = get_aligned_frames(pipeline, align)
        
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        color_image, depth_image = rgbd_to_numpy(color_frame, aligned_depth_frame)
        

        print(color_image.shape)
        print(depth_image.shape)
        print('Hello')
        color_image_1 = color_image.copy()

        #rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        
        # Apply YOLO to this image and get bounding boxes
        hands, items= YOLO.get_item_of_interest(color_image)
        result = run(hands,items)
        
        # a = items[np.unravel_index(result.argmax(), result.shape)[1]]
        hand_centroids = []

        if len(hands) > 0 and result.shape[1] != 0:
            hand_of_interest = hands[np.unravel_index(result.argmax(), result.shape)[0]]
        
            if hand_of_interest is not None:
                x1, y1, x2, y2 = hand_of_interest
                depth_crop = depth_image[y1:y2, x1:x2]

                median_depth_hand = np.median(depth_crop.reshape(-1))
                hand_centroids.append((median_depth_hand, hand_of_interest))
                cv2.putText(color_image,'Depth:' + str(median_depth_hand), 
                                (hand_of_interest[0],hand_of_interest[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                               1,
                                (255,255,255),
                                2)
            cv2.rectangle(color_image,(hand_of_interest[0],hand_of_interest[1]),(hand_of_interest[2],hand_of_interest[3]),(0,255,0),2)
        item_centroids = []
             
        for box in items: 
            if box is not None:
                x1, y1, x2, y2 = box
                depth_crop = depth_image[y1:y2, x1:x2]
                median_depth_item = np.median(depth_crop.reshape(-1))
                item_centroids.append((median_depth_item, box))
                #depth = depth_image[x_center, y_center]
                #item_boxes.append(box)
                #item_centroids.append((x_center, y_center, depth))
                cv2.putText(color_image,'Depth:' + str(median_depth_item), 
                                (box[0],box[1]), cv2.FONT_HERSHEY_SIMPLEX, 
                                1,
                                (255,255,255),
                                2)
                cv2.rectangle(color_image,(box[0],box[1]),(box[2],box[3]),(0,0,255),4)
        
        # All hands and items are detected
        
        if (len(hand_centroids) != 0) and (len(item_centroids) != 0):
            hand_depth = hand_centroids[0][0]
            hand_box = hand_centroids[0][1]
            diffs = []
            print("Item Centroids: ", item_centroids)
            for item in item_centroids:
                item_depth = item[0]
                item_box = item[1]
                print("Hand Box: ", hand_box)
                print("Item Box: ", item_box)
                iou = run(np.array([hand_box]), np.array([item_box]))
                print("IOU :", iou)
                if iou > 0:
                    depth_diff = abs(hand_depth - item_depth)
                    diffs.append((item_box, depth_diff))

            item_box = sorted(diffs, key=lambda x: x[1])
            print("Item Box: ", item_box)
            if len(item_box) > 0:
                final_tuple = item_box[0]
                depth = final_tuple[1]
                if depth < 150:
                    final_box = final_tuple[0]
                    print("Final Box: ", final_box)
                    cv2.rectangle(color_image_1,(final_box[0],final_box[1]),(final_box[2], final_box[3]),(0,0,0),2)        
        
        #Get centroid of each bounding box


        #cv2.imshow('Color',rgb_image)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image', 1280,720)
        cv2.imshow('image',color_image)
        cv2.waitKey(1)

        #cv2.imshow('Imp',color_image_1)
        cv2.namedWindow('ioi', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('ioi',1280, 720)
        cv2.imshow('ioi',color_image_1)
        cv2.waitKey(1)

finally:
    pipeline.stop()