import cv2
import numpy as np
import pyrealsense2 as rs
import glob
import os
from depth_smoothing.smooth import Smoother
from Calibration.read_cameras import Realsense


def plot_boxes(mask, image, area_thresh=3000):

    im2, ctrs, hier = cv2.findContours(
        mask[:, :, 0].copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, ctr in enumerate(ctrs):
        if cv2.contourArea(ctr) > area_thresh:
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return image


def main():
    path = "/media/veeve/46ec8a8e-15a3-421c-a2bd-69f62685f30e/Data/kiosk-training-2/077326159019-1553217903/*bag"
    backgrounds = glob.glob("*jpg")
    backgrounds = [i.split(".")[0] for i in backgrounds]
    bags = glob.glob(path)[0]

    print (bags)
    backgrounds = [cv2.imread(i+".jpg", 0)
                   for i in backgrounds if i.split("_")[0] in bags][0]

    # ctx = rs.context()
    # devices_list = []
    # for d in ctx.devices:
    #     if d.get_info(rs.camera_info.name) == "Intel RealSense D435":
    #         devices_list.append(d.get_info(rs.camera_info.serial_number))

    # cameras = [Realsense(i, (640, 360), (848, 480), 30, False, False)
    #            for i in devices_list]
    camera = Realsense(bags, (640, 360), (848, 480), 30, True, True)
    # poly = np.array(
    #     [[106, 309], [340, 200], [847, 429], [847, 478], [188, 473]])
    poly = np.array(
        [[0, 0], [847, 0], [847, 479], [0, 478]])
    i = 0
    plane = np.zeros((480, 848))
    plane = backgrounds
    print (plane.dtype)
    plane_mask = np.zeros((480, 848, 3))
    kernel = np.ones((5, 5), np.uint8)

    while True:
        segmentation_mask = np.zeros((480, 848, 3))
        rgb, depth = next(camera.get_frames())

        depth = Smoother().smooth(depth, 1400, 180)
        if i > 2 and i < 10:
            pass
            # plane += depth
        if i == 10:
            # plane /= 7
            cv2.fillPoly(plane_mask, pts=[poly], color=(1, 1, 1))
            plane_mask[(plane_mask[:, :, 0] !=
                        1) * (plane_mask[:, :, 1] != 1)] = 0
        i += 1
        plane_points = np.where(rgb[:, :, 0] == 155)

        resultant_img = plane-depth
        resultant_img *= plane_mask[:, :, 0].astype("uint8")
        output = rgb.copy().astype(float)
        segmentation_mask[resultant_img > 2] = 255
        segmentation_mask = cv2.morphologyEx(
            segmentation_mask, cv2.MORPH_OPEN, kernel, iterations=3)

        output = plot_boxes(segmentation_mask.astype("uint8"),
                            output.astype("uint8")).astype(float)
        segmentation_mask = np.where(segmentation_mask == 255)
        output[segmentation_mask[0], segmentation_mask[1], 1] *= 3
        output[output > 255] = 255

        # output[segmentation_mask == 255] = 255
        # output[depth < 10] = rgb[depth < 10]
        path = "outputs/"+str(i)+".jpg"
        cv2.imwrite(path, output.astype("uint8"))

        # if not cameras[0].show_image(output.astype("uint8")):
        # break

main()
