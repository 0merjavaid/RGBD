
import numpy as np
import cv2
from read_cameras import Realsense
import pyrealsense2 as rs

class Calibrate:

    def __init__(self, points):
        self.points = points
        self.colors = [[0, 0, 255] for i in range(len(self.points))]

    def plot_circles(self, frame):
        
        for point, color in zip(self.points, self.colors):
            cv2.circle(frame, (point[0], point[1]), 2, color, -1)
            cv2.circle(frame, (point[0], point[1]), 15, color, 2)

    def diff(self, points, medians, thres=5):
        diff = np.abs((np.array(points)-np.array(medians)))
        print ("difference is:", diff)
        for i in range(len(self.points)):
            if diff[i] < thres:
                self.colors[i] = [0, 255, 0]
            else:
                self.colors[i] = [0, 0, 255]



def main():
    #cam 3 1008 927
    #cam 1 1138 982
    #cam 4 1120 895
    #cam 5 1009 902

    points=np.loadtxt("points.txt").reshape(-1,5)
    median_depths=np.loadtxt("depths.txt").reshape(-1,3)
    cam_ids=points[:,0]
    points=points[:,1:].reshape(-1,2,2).astype(int)
    cam_ids=points[:,0]
    median_depths=median_depths[:,1:].reshape(-1,2).astype(int)
     
    ctx = rs.context()
    devices_list = []
    for d in ctx.devices:
        if d.get_info(rs.camera_info.name) == "Intel RealSense D435":
            devices_list.append(d.get_info(rs.camera_info.serial_number))
    print (len(devices_list), " Cameras Connected")
    cameras=[Realsense(i,(640, 360), (848, 480), 30) for i in devices_list]
     
    for camera in cameras:
        camera.assign_cam_id()
    
    
    for camera in cameras: 
        point=points[camera.cam_id-1]

        print ("cam id", camera.cam_id, point)
        calibrator = Calibrate(point)
        median = [0, 0]
        stored_depths = median_depths[camera.cam_id-1]
        ret = True
        while ret:
        
            color_frame, depth_frame = next(camera.get_frames())
            for i in range(len(point)):
                print (i)
                median[i] = np.median(
                    depth_frame[point[i][1]-1:point[i][1]+1, point[i][0]-1:point[i][0]+1])
            print ("median depth", median)
            calibrator.diff(stored_depths, median, 15)
            calibrator.plot_circles(color_frame)
            cv2.imshow("", color_frame)
            key=cv2.waitKey(1)
            if key==ord("q"):
                break
        cv2.destroyAllWindows()

main()
