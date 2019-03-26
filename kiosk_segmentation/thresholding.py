import cv2
import numpy as np
import pyrealsense2 as rs
import glob
import os
from depth_smoothing.smooth import Smoother
from Calibration.read_cameras import Realsense


def main():
    path = "/home/veeve/Workspace/data-aquisition-tool/python/Data/version1/kiosk_ioi/012044040058-1551429365/*bag"
    # bags = glob.glob(path)[0]
    ctx = rs.context()
    devices_list = []
    for d in ctx.devices:
        if d.get_info(rs.camera_info.name) == "Intel RealSense D435":
            devices_list.append(d.get_info(rs.camera_info.serial_number))

    cameras = [Realsense(i, (640, 360), (848, 480), 30, False, False)
               for i in devices_list]
    # camera = Realsense(bags, (640, 360), (848, 480), 30, True, True)
    poly = np.array(
        [[106, 309], [340, 200], [847, 429], [847, 478], [188, 473]])
    i = 0
    plane = np.zeros((480, 848))
    plane_mask = np.zeros((480, 848, 3))
    kernel = np.ones((5, 5), np.uint8)

    while True:
        segmentation_mask = np.zeros((480, 848, 3))
        rgb, depth = next(cameras[1].get_frames())

        depth = Smoother().smooth(depth, 1400, 180)
        if i > 2 and i < 10:
            plane += depth
        if i == 10:
            print (i)
            plane /= 7
            cv2.fillPoly(plane_mask, pts=[poly], color=(1, 1, 1))
            plane_mask[(plane_mask[:, :, 0] !=
                        1) * (plane_mask[:, :, 1] != 1)] = 0
        i += 1

        plane_points = np.where(rgb[:, :, 0] == 155)

        resultant_img = plane-depth
        resultant_img *= plane_mask[:, :, 0]
        output = rgb.copy()
        segmentation_mask[resultant_img > 2] = 255
        segmentation_mask = cv2.morphologyEx(
            segmentation_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        output[segmentation_mask == 255] = 255
        output[depth < 10] = rgb[depth < 10]

        if not cameras[0].show_image(output.astype("uint8")):
            break

main()
