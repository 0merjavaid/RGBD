import cv2
import numpy as np


class DepthSegmentation:

    def __init__(self, initial_frame, boundary_points=None):
        """
        initial_frame: np array of Empty frame of Kiosk or Cart
        boundary_points: list or numpy of four points(base of cart or kiosk) in clockwise order
        """
        assert isinstance(initial_frame, np.ndarray)
        self.initial_frame = initial_frame
        self.boundary_points = boundary_points
        self.get_base_depth()
        self.base_mask = self.get_base_depth()

    def get_base_depth(self):
        base_mask = np.ones_like(self.initial_frame)
        if self.boundary_points is not None:
            base_mask = np.zeros_like(self.initial_frame)
            cv2.fillConvexPoly(base_mask, self.boundary_points, 1)
        self.initial_frame = self.smooth(self.initial_frame).astype("uint8")
        return base_mask

    def smooth(self, depth_image, max_depth=800, min_depth=180):

        depth_image[depth_image > max_depth] = max_depth
        depth_image[depth_image < min_depth] = min_depth
        depth_image -= min_depth
        depth_image = ((depth_image/(max_depth-min_depth))*255).astype("uint8")
        kernel = np.array(
            [-1, -1, -1, -1, 9, -1, -1, -1, -1]).reshape(3, 3)
        depth_image = cv2.filter2D(depth_image, -1, kernel)

        return depth_image

    def get_segmask(self, depth_image):
        """
        depth_image: np array of depth image with 1 channel

        returns: segmentation_mask, binary segmenation mask with items as 
                                    forgorunds (255) and kiosk as bg(0)
                 contours,          list of contours for items
        """
        kernel = np.ones((5, 5), np.uint8)
        segmentation_mask = np.zeros_like(depth_image)
        depth_image = self.smooth(depth_image)
        difference = self.initial_frame - depth_image
        segmentation_mask[difference > 0.5] = 255
        segmentation_mask[difference <= 0.5] = 0
        segmentation_mask = cv2.morphologyEx(
            segmentation_mask, cv2.MORPH_OPEN, kernel, iterations=9)\
            * self.base_mask
        contours = self.get_contours(segmentation_mask)
        return segmentation_mask, contours

    def get_contours(self, mask, min_area=100):
        mask[mask > 30] = 255
        mask[mask <= 30] = 0
        contours, _ = cv2.findContours(mask.astype(
            "uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        assert len(contours) > 1
        contours = [i for i in contours if cv2.contourArea(i) > min_area]
        return contours
