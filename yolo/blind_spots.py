import pyrealsense2 as rs
import numpy as np
import cv2
import time
import threading
from multiprocessing import Pool


Frames_list = []
flag = True
class Threadshow(threading.Thread):

    def __init__(self, ncams):
        self.ncams = ncams
        threading.Thread.__init__(self)

    def run(self):
        while flag:
            img = video_frame(self.ncams)
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cv2.imshow("Frame", img)
            except:
                pass
            cv2.waitKey(1)


class CameraThread(threading.Thread):
    def __init__(self, i, device):
        """
        Constructor of camera thread which captures all
        frames from different cameras
        :param cap: opencv capture variable to capture frame from camera i
        :param i: the camera index
        :param out: the outstream for writing the frames
        :param com_var: the variable used to share data among threads
        """
        
        self.index = i
        
        self.pipeline= device
        threading.Thread.__init__(self)

    def run(self):
        i = 0
        '''pipeline_1 = rs.pipeline()
                                config_1 = rs.config()
                                config_1.enable_device(self.device)
                                config_1.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                                config_1.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
                                pipeline_1.start(config_1)'''
        start = time.time()
        #align_to = rs.stream.color
        #align = rs.align(align_to)
        global flag
        while 1:
            start1=time.time()
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            frame = np.asanyarray(color_frame.get_data())
            Frames_list[self.index] = frame
            done = time.time()
            # if(done - start >= 10):
            #   break
            i += 1
        flag = False

def video_frame(ncams):
    """
    Concat different frames to make a displaying image
    :return: image to be displayed on UI
    """
    try:
        global Frames_list
        if len(Frames_list) % 2 == 1:
            img = np.zeros((720, 1280, 3))
            img[img == 0] = 255
            img = img.astype('uint8')
            Frames_list.append(img)
        final_img = cv2.hconcat([Frames_list[0],
                                 Frames_list[1]])
        i = 2
        while i < len(Frames_list):
            image = cv2.hconcat([Frames_list[i],
                                 Frames_list[i + 1]])
            final_img = cv2.vconcat([final_img, image])
            i += 2
        final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        screensize = int((ncams + 1) / 2)
        #final_img = cv2.resize(final_img, (screensize * 640, screensize * 360))
        return final_img
    except:
        pass

camthread = []
pipeline = []
list_devices = []
config = []
ctx = rs.context()
k = 2
for d in ctx.devices:
    if d.get_info(rs.camera_info.name) == "Intel RealSense D435":
        list_devices.append(d.get_info(rs.camera_info.serial_number))
for i in range(k):
    print(i)
    pipeline_1 = rs.pipeline()
    config_1 = rs.config()
    config_1.enable_device(list_devices[i])
    config_1.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config_1.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    pipeline_1.start(config_1)
    config.append(config_1)
    pipeline.append(pipeline_1)
for i in range(k):
    cam = CameraThread(i, pipeline[i])
    Frames_list.append('')
    camthread.append(cam)
    #camthread[i].start()
for i in range(k):
    camthread[i].start()
show = Threadshow(k)
show.start()
