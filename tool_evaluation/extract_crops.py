import numpy as np
import cv2

import yolo
from depth_utils import *
from utils import *


# Setting GPU device
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

# Hyperparameters
batch_size = 1
img_size = 256
ioi_width_thres = 50
weights_path = 'weights/UNET_5.pt'

# Get transforms
print('Loading transforms...')
transform = get_transforms(img_size)

# Get model
print('Loading model...')
seg_model = get_segmentation_model(weights_path)
seg_model.eval()

print('Loading YOLO')
YOLO = yolo.YOLO()


def get_ioi_bbox_depth(color_image, depth_data, transform, seg_model):
    bbox = None
    hand_box = None
    original_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    seg_mask = segment_hand(color_image, transform, seg_model)
    mask = seg_mask.copy()
    contours = get_contours(seg_mask)  # ~1 millisecond
    bounding_box, hand_contour = refine_contours(contours)  # ~0.1 millisecond
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
            ioi_mask = fill_ioi(color_image, ioi)
            left_black_max, right_black_min = width_points(
                ioi_mask[:, :, 0], center_x, hand_bottom_y)
            if left_black_max is not None and right_black_min is not None:
                ioi_mask[:, :(left_black_max - ioi_width_thres)] = 0
                ioi_mask[:, (right_black_min + ioi_width_thres):] = 0

                y_threshold = get_multiple_widths(
                    ioi_mask[:, :, 0], center_x, hand_bottom_y, ioi_height)
                ioi_mask[y_threshold:, :] = 0

            ioi_mask_coorinates = np.where(ioi_mask == 1)
            hand_box = np.array(bounding_box)
            if 1 in np.unique(ioi_mask):

                x_min = min(ioi_mask_coorinates[1])
                x_max = max(ioi_mask_coorinates[1])
                y_max = max(ioi_mask_coorinates[0])
                if (y_max - hand_box[3] > 50):
                    bbox = np.array([x_min, center_y, x_max, y_max])

    return bbox, hand_box


def get_ioi_bbox_yolo(color_image, depth_image):
    final_box = None
    color_image_ioi = color_image.copy()

    # Apply YOLO to this image and get bounding boxes
    hands, items = YOLO.get_item_of_interest(color_image)
    result = box_iou(hands, items)

    # get hand centroid
    hand_centroids = []
    if len(hands) > 0 and result.shape[1] != 0:
        hand_of_interest = get_hand_of_interest(hands, result)
        if hand_of_interest is not None:
            median_depth_hand = get_median(hand_of_interest, depth_image)
            hand_centroids.append((median_depth_hand, hand_of_interest))
            #color_image = draw_boundingBox(color_image, hand_of_interest, str('Depth: ' + str(median_depth_hand)))

    # Get Item centroid
    item_centroids = []
    for box in items:
        if box is not None:
            median_depth_item = get_median(box, depth_image)
            item_centroids.append((median_depth_item, box))
            #color_image = draw_boundingBox(color_image, box, str('Depth: ' + str(median_depth_item)), box_color=(255,0,0))

    # All hands and items are detected
    if (len(hand_centroids) != 0) and (len(item_centroids) != 0):
        final_box = get_item_of_interest(
            hand_centroids, item_centroids, threshold=10000)
        if final_box is not None:
            return final_box
            #color_image = draw_boundingBox(color_image_ioi, final_box, box_color=(0,0,0), box_thickness=4)
    return final_box


def save_crops(color_image, depth_image, method='yolo'):
    ioi_bbox = None
    hand_box = None
    if method == 'yolo':
        ioi_bbox = get_ioi_bbox_yolo(color_image, depth_image)
    else:
        ioi_bbox, hand_box = get_ioi_bbox_depth(
            color_image, depth_image, transform, seg_model)
    return ioi_bbox, hand_box
