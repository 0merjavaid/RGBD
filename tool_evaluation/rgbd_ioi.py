# -*- coding: utf-8 -*-
"""
Created on Wed Nov 7 15:00:00 2018
@author: Haroon Rashid (haroon.rashid@visionx.io)
"""
from smartcart import messagetypes as gi
from .Unet import unet_inference
from .Unet.utils import *


class RgbdIoI:

    def __init__(self, model_path, img_size, batch_size):
        self._unet_object = unet_inference.UnetInference(
            model_path, img_size, batch_size)

    def _extract_ioi(self, req):
        assert isinstance(req, gi.message_objects.RgbdImage)

        res = gi.message_objects.IoIResponse()
        res.img_id = req.img_id
        res.cam_id = req.cam_id
        res.timestamp = req.timestamp
        res.reset_tracker = False
        res.weight = req.weight
        res.num_objs = 1
        res.rgb_data = cv2.cvtColor(req.rgb_data, cv2.COLOR_BGR2RGB)
        res.bounding_boxes = np.array([0, 0, 0, 0])
        res.hand_box = np.array([0, 0, 0, 0])

        rgb_data = req.rgb_data
        depth_data = req.depth_data
        hands, seg_mask = self._unet_object.get_hands(rgb_data, 1000)

        seg_mask[seg_mask > 0.5] = 1
        seg_mask[seg_mask <= 0.5] = 0
        mask = seg_mask.copy()

        contours = get_contours(seg_mask)

        bounding_box, hand_contour = refine_contours(contours)
        if hand_contour is not None:
            mask_ioi = get_mask_ioi(depth_data, mask, 90, hand_contour)

            ioi_candidates = get_contours(mask_ioi * 255)
            ioi_width_thres = 50
            center_x, center_y = get_bbox_center(bounding_box)
            hand_bottom_y = int(np.max(hand_contour.reshape(-1, 2)[:, 1]))
            ioi = get_filtered_ioi(ioi_candidates, (center_x, center_y))
            if ioi is not None:
                ioi = ioi.reshape(-1, 2)
                ioi_height = np.max(ioi[:, 1]) - hand_bottom_y
                ioi_mask = fill_ioi(rgb_data, ioi)
                left_black_max, right_black_min = width_points(
                    ioi_mask[:, :, 0], center_x, hand_bottom_y)
                if left_black_max is not None and right_black_min is not None:
                    ioi_mask[:, :(left_black_max - ioi_width_thres)] = 0
                    ioi_mask[:, (right_black_min + ioi_width_thres):] = 0

                    y_threshold = get_multiple_widths(
                        ioi_mask[:, :, 0], center_x, hand_bottom_y, ioi_height)
                    ioi_mask[y_threshold:, :] = 0

                ioi_mask_coorinates = np.where(ioi_mask == 1)
                res.hand_box = np.array(bounding_box)
                if 1 in np.unique(ioi_mask):

                    x_min = min(ioi_mask_coorinates[1])
                    x_max = max(ioi_mask_coorinates[1])
                    y_max = max(ioi_mask_coorinates[0])
                    res.bounding_boxes = np.array([x_min, center_y, x_max, y_max])
                res.masked_rgb = ioi_mask
        return res
