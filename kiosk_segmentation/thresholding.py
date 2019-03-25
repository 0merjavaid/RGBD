import cv2
import numpy as np
import pyrealsense2 as rs
import glob
import os
from depth_smoothing.smooth import Smoother
from Calibration.read_cameras import Realsense


def main():
    path = "/home/veeve/Workspace/data-aquisition-tool/python/Data/version1/kiosk_ioi/012044040058-1551429365/*bag"
    bags = glob.glob(path)[0]
    camera = Realsense(bags, (640, 360), (848, 480), 30, True, True)
    while True:
        rgb, depth = next(camera.get_frames())
        if not camera.show_image(rgb):
            break

main()
