
import numpy as np
import cv2
from read_cameras import Realsense


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
    points = [[160, 100], [500, 160]]
    realsense = Realsense((640, 360), (848, 480), 30)
    realsense.assign_cam_id()
    return
    calibrator = Calibrate(points)
    median = [0, 0]
    stored_depths = [366, 408]
    ret = True
    while ret:
        color_frame, depth_frame = next(realsense.get_frames())
        for i in range(len(points)):
            print (i)
            median[i] = np.median(
                depth_frame[points[i][1]-1:points[i][1]+1, points[i][0]-1:points[i][0]+1])
        print ("median depth", median)
        calibrator.diff(stored_depths, median, 5)
        calibrator.plot_circles(color_frame)
        cv2.imshow("", color_frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()

main()
