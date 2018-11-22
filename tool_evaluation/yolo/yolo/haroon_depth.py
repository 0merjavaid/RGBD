import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import yolo for object detection
import yolo
from operator import itemgetter

# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

frames = pipeline.wait_for_frames()
# frames.get_depth_frame() is a 640x360 depth image
aligned_frames = align.process(frames)

# Get aligned frames
aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
print('Aligned Depth Frame')
print(aligned_depth_frame)
color_frame = aligned_frames.get_color_frame()

depth_image = np.asanyarray(aligned_depth_frame.get_data())
color_image = np.asanyarray(color_frame.get_data())

# Init YOLO model
YOLO=yolo.YOLO()

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

# def get_min_dists(hand_centroids, item_centroids):
#     print('Hand Centroids:')
#     print(hand_centroids)
#     print('item_centroids')
#     print(item_centroids)
#     distances = []
#     for hand_centroid in hand_centroids:
#         distance_hand = []
#         p1 = hand_centroid[2]
#         for idx, centroid in enumerate(item_centroids):
#             p2 = centroid[2]
#             dist = abs(p2 - p1)
#             #dist = np.sqrt(squared_dist)
#             print('dist')
#             print(dist)
#             distance_hand.append(dist)
#             print('distance hand')
#             print(distance_hand)
#         distances.append(distance_hand)
#     distances = np.array(distances)
#     print('distances')
#     print(distances.shape)
#     print(distances)
#     min_distance = np.amin(distances,axis=1,keepdims=True)
#     min_index = np.argmin(distances,axis=1)

#     return min_index, min_distance

# def get_min_dist(hand_centroids, item_centroids):
#     distances = []
#     hand_centroid = hand_centroids[0]
#     p1 = np.int16(hand_centroid[2])
#     print(type(p1))
#     for idx, centroid in enumerate(item_centroids):
#         p2 = np.int16(centroid[2])
#         print(p2,p1)
#         print(p2 - p1)
#         print(type(p1))
#         dist = abs(p2 - p1)
#         distances.append([idx, dist])
#     print('Unsoreted distances')
#     print(distances)
#     distances = sorted(distances, key=itemgetter(1))
#     print('Sorted Distances')
#     print(distances)
#     val = distances[0]
#     return val[0], val[1]

try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image
        
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)
        
        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()


        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue
        
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
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