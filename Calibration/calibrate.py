
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


def get_regions_depth(depth_frame, plane_points, plane_height, col_intervals):
    cols = int(depth_frame.shape[1]/col_intervals)
    cols_depths = np.zeros(
        (len(plane_points), col_intervals, plane_height, cols))
    non_zeros = np.zeros((len(plane_points), col_intervals))

    for i in range(len(plane_points)):

        cols_depths[i] =\
            np.array([depth_frame[plane_points[i]:plane_points[
                     i]+plane_height, cols*j:cols*(j+1)]
                for j in range(col_intervals)]).squeeze()

        non_zeros[i] = np.array([np.count_nonzero(cols_depths[i, j])
                                 for j in range(col_intervals)])

    cols_depths = np.sum(
        cols_depths.reshape(len(plane_points), col_intervals, -1), -1)/non_zeros

    return cols_depths


def mask_inturrept(image, indexes, box_height, box_width):

    for i in indexes[0]:
        for j in indexes[1]:
            y1, y2, x1, x2 = i * \
                box_height, (i*box_height)+box_height, j * \
                box_width, (j*box_width)+box_width
            print (y1, y2, x1, x2)
            image[y1:y2, x1:x2] = [0, 255, 0]
    return image


def detect_inturrept(depth_frame, buffer_counter, buffer_depths):
    col_intervals = 10
    row_intervals = 2
    cols_boundary = np.zeros((col_intervals,))
    cols = int(848/col_intervals)
    rows = int(480/row_intervals)
    rows = 100
    thres = 10
    plane_points = [100]
    plane_height = 5

    depth_frame = (depth_frame.astype(float)*255/1000).astype("uint8")
    depth_frame = cv2.GaussianBlur(depth_frame, (5, 5), 0)
    cols_depths = get_regions_depth(
        depth_frame, plane_points, plane_height, col_intervals)

    if buffer_counter < 10:
        buffer_depths = np.array(buffer_depths)
        buffer_depths = np.concatenate(
            (cols_depths.reshape(1, row_intervals-1, -1), buffer_depths[0:-1, ]), axis=0)

    cols_boundary = np.mean(buffer_depths, 0)

    diffs = abs(cols_depths-cols_boundary)
    difference_detected = np.where(diffs > thres)

    if len(difference_detected[0]) > 0:
        return True, buffer_counter, buffer_depths
    else:
        return False, buffer_counter, buffer_depths
        # mask_inturrept(color_frame, difference_detected, rows, cols)


def plane_inturrept():
    cam_name = rs.context().devices[0].get_info(
        rs.camera_info.serial_number)
    print (cam_name)
    camera = Realsense(cam_name, (640, 360), (848, 480), 30)

    buffer_counter = 0

    # [rows*(i+1) for i in range(row_intervals-1)]

    buffer_size = 3

    buffer_depths = [np.zeros((buffer_size, 1,  10)) for i in range(cams))]
    while True:
        buffer_counter += 1
        color_frame, depth_frame=next(camera.get_frames())
        depth_frame[depth_frame > 1000]=1000
        _, __, buffer_depths=detect_inturrept(
            depth_frame, buffer_counter, buffer_depths)
        print _

        # cv2.imshow("", color_frame)
        # key = cv2.waitKey(1)
        # if key == ord("q"):
        #     break
        # cv2.destroyAllWindows()


def main():

    points=np.loadtxt("points.txt").reshape(-1, 5)
    median_depths=np.loadtxt("depths.txt").reshape(-1, 3)
    cam_ids=points[:, 0]
    points=points[:, 1:].reshape(-1, 2, 2).astype(int)
    cam_ids=points[:, 0]
    median_depths=median_depths[:, 1:].reshape(-1, 2).astype(int)

    ctx=rs.context()
    devices_list=[]
    for d in ctx.devices:
        if d.get_info(rs.camera_info.name) == "Intel RealSense D435":
            devices_list.append(d.get_info(rs.camera_info.serial_number))
    print (len(devices_list), " Cameras Connected")
    print (devices_list)
    cameras=[Realsense(i, (640, 360), (848, 480), 30) for i in devices_list]

    for camera in cameras:
        camera.assign_cam_id()

    for camera in cameras:
        point=points[camera.cam_id-1]

        calibrator=Calibrate(point)
        median=[0, 0]
        stored_depths=median_depths[camera.cam_id-1]
        ret=True
        while ret:

            color_frame, depth_frame=next(camera.get_frames())
            for i in range(len(point)):
                # print (i)
                median[i]=np.median(
                    depth_frame[point[i][1]-1:point[i][1]+1,
                                point[i][0]-1:point[i][0]+1])
            # print ("median depth", median)

            calibrator.diff(stored_depths, median, 15)
            calibrator.plot_circles(color_frame)
            cv2.imshow("", color_frame)
            key=cv2.waitKey(1)
            if key == ord("q"):
                break
        cv2.destroyAllWindows()

# main()
plane_inturrept()
