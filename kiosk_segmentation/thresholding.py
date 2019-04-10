import cv2
import numpy as np
import pyrealsense2 as rs
import glob
import os
from depth_smoothing.smooth import Smoother
from Calibration.read_cameras import Realsense
import sys
mouse_points = list()


def plot_boxes(mask, image, area_thresh=3000):

    im2, ctrs, hier = cv2.findContours(
        mask[:, :, 0].copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, ctr in enumerate(ctrs):
        if cv2.contourArea(ctr) > area_thresh:
            x, y, w, h = cv2.boundingRect(ctr)
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 3)
    return image


def draw_circle(event, x, y, flags, param):
    mouseX, mouseY = None, None
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX, mouseY = x, y
        global mouse_points
        mouse_points.append([mouseX, mouseY])
        print (mouse_points)
        return param


def set_base_boundary(images_folder, bag=False, number_of_cameras=4):
    if bag:
        pass
    else:
        files = os.listdir(images_folder)
        cameras = list(set([i.split("_")[-1].split(".")[0][:12]
                            for i in files if "png" in i]))
        print (cameras)

        i = 0

        for file in files:
            global mouse_points

            file = images_folder+"/"+file
            if i == number_of_cameras:
                break
            if cameras[i] in file and "png" in file:
                img = cv2.imread(file)
                cv2.namedWindow('image')
                cv2.setMouseCallback(
                    'image', draw_circle)
                print (file)
                cv2.imshow('image', img)
                key = cv2.waitKey()
                if key == ord("q") or len(mouse_points) == 4:
                    np.savetxt("kiosk_segmentation/"+cameras[i]+"txt",
                               np.array(mouse_points).reshape(-1, 2))
                    mouse_points = list()
                    i += 1


def main():
    path = "../depth_seg_test/"

    # set_base_boundary(path, False, 3)
    folders = os.listdir(path)
    camera_points = glob.glob("kiosk_segmentation/*txt")
    cameras = [i.split(".")[0].split("/")[-1] for i in camera_points]
    stds = list()
    for folder in folders:
        if "-" not in folder:
            continue
        camera_pngs, camera_depths, camera_boundaries = dict(), dict(), dict()
        pngs = sorted(glob.glob(os.path.join(path, folder)+"/*png"),
                      key=lambda x: int(x.split("/")[-1].split("_")[0]))
        depths = sorted(glob.glob(os.path.join(path, folder)+"/*npy"),
                        key=lambda x: int(x.split("/")[-1].split("_")[0]))

        for i in range(len(cameras)):

            camera_pngs[cameras[i]] = [
                j for j in pngs if cameras[i] in j]
            camera_depths[cameras[i]] = [
                j for j in depths if cameras[i] in j]
            camera_boundaries[cameras[i]] = [np.loadtxt(
                camera_points[j]).reshape(-1, 2) for j in range(len(camera_points)) if cameras[i] in camera_points[j]]

        for key in camera_pngs.keys():
            lis = list()
            plane = np.zeros((480, 848))
            plane_mask = np.zeros((480, 848, 3))
            kernel = np.ones((5, 5), np.uint8)
            for rgb, depth in zip(camera_pngs[key], camera_depths[key]):
                boundary = camera_boundaries[key]
                base = plane.copy().astype('uint8')
                boundary = np.int32(boundary)

                cv2.fillConvexPoly(base, boundary, 1)
                segmentation_mask = np.zeros((480, 848))
                frame_no = int(rgb.split("/")[-1].split("_")[0])
                depth, rgb_image = np.load(depth), cv2.imread(rgb)
                depth = Smoother().smooth(depth, 800, 180)

                if frame_no > 2 and frame_no < 10:
                    lis.append(depth)
                    continue
                    # plane += depth
                if frame_no == 13:
                    # plane /= 7
                    plane_mask = depth
                    continue
                    # cv2.fillPoly(plane_mask, pts=[poly], color=(1, 1, 1))
                    # plane_mask[(plane_mask[:, :, 0] !=

                # plane_points = np.where(rgb[:, :, 0] == 155)
                if frame_no < 30:
                    continue
                stds.append(np.mean(np.std(np.array(lis), 0)))

                resultant_img = plane_mask-depth
                # resultant_img *= plane_mask[:, :, 0].astype("uint8")
                output = rgb_image.copy().astype(float)
                segmentation_mask[resultant_img > 0.51] = 255
                segmentation_mask = cv2.morphologyEx(
                    segmentation_mask, cv2.MORPH_OPEN, kernel, iterations=9)*base
                white_mask = segmentation_mask.copy()
                # continue

                # output = plot_boxes(segmentation_mask.astype("uint8"),
                #                     output.astype("uint8")).astype(float)
                segmentation_mask = np.where(segmentation_mask == 255)
                output[segmentation_mask[0], segmentation_mask[1], 1] *= 3
                output[output > 255] = 255
                # output[segmentation_mask == 255] = 255
                # output[depth < 10] = rgb[depth < 10]
                save_path = "outputs/" + "seg_" + \
                    "_".join(rgb.split("/")[2:])

                # cv2.imshow("", output.astype("uint8"))
                # cv2.imshow("s", base.astype("uint8"))
                # key = cv2.waitKey()
                # if key == ord("q"):
                #     sys.exit()
                #     break
                # # print (save_path)
                print (cv2.imwrite(save_path, white_mask.astype("uint8")))

            # if not cameras[0].show_image(output.astype("uint8")):
            # break
    print (np.mean(stds))

main()

aa = """
2
2
2
2
1
1
1
1
2
3
2
4
2
1
1
1
1
2
1
1
1
1
1
1
1
2
2
1
1
1
2
1
1
2
1
1
1
2
1
2
2
1
1
2
1
1
1
1
1
1
2
2
1
2
1
2
1
3
1
3
1
"""
sum([int(i) for i in aa.split()])
