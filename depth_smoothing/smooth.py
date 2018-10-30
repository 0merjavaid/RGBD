import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import yolo for object detection
#import yolo
from scipy import ndimage

from operator import itemgetter
import time

# Create a pipeline
print('Creating Pipeline...')
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
print('Configuring Pipeline...')
config = rs.config()
#config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
#config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)


# Start streaming
print('Starting Stream...')
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 10 #1 meter
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

fgbg = cv2.createBackgroundSubtractorMOG2()

 

try:
    while True:
        start = time.time()
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

        end = time.time()
        
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
 
        max_depth=1000
        min_depth=180
        depth_image[depth_image>max_depth]=max_depth
        depth_image[depth_image<min_depth]=min_depth
        depth_image-=min_depth

        depth_image=((depth_image/(max_depth-min_depth))*255).astype("uint8")
        #depth_image1=cv2.medianBlur(depth_image,5)
        
        #depth_image1 = ndimage.grey_dilation(depth_image, size=(5, 5), structure=np.ones((5, 5)))
        start_time=time.time()
        depth_image1 = cv2.equalizeHist(depth_image)
        depth_image1 = ndimage.grey_dilation(depth_image1, size=(5, 5), structure=np.ones((5, 5)))
        depth_image1=cv2.medianBlur(depth_image1,5)
        print (time.time()-start_time)
        kernel=np.array([-1,-1,-1,-1,9,-1,-1,-1,-1]).reshape(3,3)
        
        depth_image1=cv2.filter2D(depth_image1,-1,kernel)
        

        depth_image1 = cv2.applyColorMap(depth_image1, cv2.COLORMAP_JET)
        depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

        compare=np.hstack((depth_image1,depth_image))
        cv2.imshow('depth_colormap', compare)
        key=cv2.waitKey(1)
        if key==ord("q"):
            break
    
    cv2.destroyAllWindows()

finally:
    pipeline.stop()


