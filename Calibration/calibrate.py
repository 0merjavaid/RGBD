
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
        # print ("difference is:", diff)
        for i in range(len(self.points)):
            if diff[i] < thres:
                self.colors[i] = [0, 255, 0]
                return True
            else:
                self.colors[i] = [0, 0, 255]
                return False


def plane_inturrept():
    cam_name = rs.context().devices[0].get_info(
        rs.camera_info.serial_number)
    print (cam_name)
    camera = Realsense(cam_name, (640, 360), (848, 480), 30)

    i = 0
    f1, s1, t1 = 0, 0, 0
    interval = 10
    quartile_boundary = np.zeros((interval,))
    quartile = int(848/interval)
    thres = 10
    plane_point = 50
    plane_height = 5
    buffer_depths = np.zeros((interval, 5))
    while True:
        i += 1
        color_frame, depth_frame = next(camera.get_frames())
        depth_frame[depth_frame > 1000] = 1000
        depth_frame = (depth_frame.astype(float)*255/1000).astype("uint8")
        depth_frame = cv2.GaussianBlur(depth_frame, (5, 5), 0)

        quartile_depths =\
            np.array([depth_frame[plane_point:plane_point+plane_height,
                                  quartile*j:quartile*(j+1)]
                      for j in range(interval)]).squeeze()

        non_zeros = np.array([np.count_nonzero(quartile_depths[j])
                              for j in range(interval)])

        quartile_depths = np.sum(
            quartile_depths.reshape(interval, -1), -1)/non_zeros

        if i < 10:
            buffer_depths = np.array(buffer_depths)
            buffer_depths = [np.concatenate(
                ([quartile_depths[j]], buffer_depths[j, 0:-1])) for j in range(interval)]

            quartile_boundary = [np.mean(buffer_depths[j])
                                 for j in range(interval)]

        diffs = np.array([np.abs(d1-d2)
                          for d1, d2 in zip(quartile_boundary, quartile_depths)])
        difference_detected = np.where(diffs > thres)

        if len(difference_detected[0]) > 0:
            pass
            print difference_detected,
            print
        depth_frame[plane_point:plane_point+plane_height] = 255
        cv2.imshow("", depth_frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


def main():

    points = np.loadtxt("points.txt").reshape(-1, 5)
    median_depths = np.loadtxt("depths.txt").reshape(-1, 3)
    cam_ids = points[:, 0]
    points = points[:, 1:].reshape(-1, 2, 2).astype(int)
    cam_ids = points[:, 0]
    median_depths = median_depths[:, 1:].reshape(-1, 2).astype(int)

    ctx = rs.context()
    devices_list = []
    for d in ctx.devices:
        if d.get_info(rs.camera_info.name) == "Intel RealSense D435":
            devices_list.append(d.get_info(rs.camera_info.serial_number))
    print (len(devices_list), " Cameras Connected")
    print (devices_list)
    cameras = [Realsense(i, (640, 360), (848, 480), 30) for i in devices_list]

    for camera in cameras:
        camera.assign_cam_id()

    for camera in cameras:
        point = points[camera.cam_id-1]

        print ("cam id", camera.cam_id, point)
        calibrator = Calibrate(point)
        median = [0, 0]
        stored_depths = median_depths[camera.cam_id-1]
        ret = True
        while ret:

            color_frame, depth_frame = next(camera.get_frames())
            for i in range(len(point)):
                # print (i)
                median[i] = np.median(
                    depth_frame[point[i][1]-1:point[i][1]+1,
                                point[i][0]-1:point[i][0]+1])
            # print ("median depth", median)

            calibrator.diff(stored_depths, median, 15)
            calibrator.plot_circles(color_frame)
            cv2.imshow("", color_frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
        cv2.destroyAllWindows()

# main()
plane_inturrept()
