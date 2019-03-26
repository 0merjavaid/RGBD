import pyrealsense2 as rs
import numpy as np
import cv2
from scipy import ndimage


class Smoother:

    def __init__(self):
        pass

    @staticmethod
    def smooth(depth_image, max_depth=1000, min_depth=180):

        depth_image[depth_image > max_depth] = max_depth
        depth_image[depth_image < min_depth] = min_depth
        depth_image -= min_depth

        depth_image = ((depth_image/(max_depth-min_depth))*255).astype("uint8")
        # depth_image1=cv2.medianBlur(depth_image,5)

        #depth_image1 = ndimage.grey_dilation(depth_image, size=(5, 5), structure=np.ones((5, 5)))
        #depth_image1 = cv2.equalizeHist(depth_image)
        depth_image1 = ndimage.grey_dilation(
            depth_image, size=(5, 5), structure=np.ones((5, 5)))
        depth_image1 = cv2.medianBlur(depth_image1, 5)

        kernel = np.array([-1, -1, -1, -1, 9, -1, -1, -1, -1]).reshape(3, 3)

        depth_image1 = cv2.filter2D(depth_image1, -1, kernel)

        smooth = cv2.applyColorMap(depth_image1, cv2.COLORMAP_JET)

        return depth_image1
