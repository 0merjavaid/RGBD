import torch
import torch.nn as nn
import time
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import pyrealsense2 as rs
import math
from unet import Unet34

RGB_IMAGE_WIDTH = 848
RGB_IMAGE_HEIGHT = 480

DEPTH_IMAGE_WIDTH = 640
DEPTH_IMAGE_HEIGHT = 360
FPS = 60


def get_base():
    """
    Base Model used for Feature Extraction
    :return: nn.Sequential layers of base model
    """
    layers = cut_model(f(True), cut)
    return nn.Sequential(*layers)


def get_segmentation_model(weights_path):
    """
    Segmentation Model used for hand segmentation
    :param weights_path: weights for trained model
    :return: model loaded with the input weights
    """
    m_base = nn.Sequential(
        *(list(models.resnet34(pretrained=False).children())[:8]))
    m = Unet34(m_base).cuda()
    m.load_state_dict(torch.load(weights_path))
    return m


def get_transforms(img_size):
    """
    Transform applied to the image at test time
    :param img_size: Reshape the image to size that model accepts
    :return: pytorch transforms to apply to image
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


def get_contours(image):
    """
    Extarct contours from a binary mask
    :param image: binary image to extract contours from
    :return contours: all contours found in the binary image 
    """
    assert len(image.shape) == 2, 'Image should be binary'
    contour_image = image.copy()
    contour_image[contour_image == 1] = 255
    _, contours, _ = cv2.findContours(contour_image.astype('uint8'),
                                      cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours


def get_contour_area(contour):
    """
    Calculate the area bounded by the contour
    :param contour: contour data points
    :return area: area of contour 
    """
    assert isinstance(contour, np.ndarray), 'contour should be a numpy array'
    return cv2.contourArea(contour)


def refine_contours(contours, area=1000):
    """
    Draw bounding box around contour
    :param image: input image to draw boxes on
    :param contours: contours around which bounding box will be drawn
    :return: (X,y,w,h) of the last contour
    """
    assert isinstance(contours, list)
    assert isinstance(area, (int, float))
    x, y, w, h = 0, 0, 0, 0
    hand_contour = None
    max_contour_dict = dict()
    for contour in contours:
        cnt_area = get_contour_area(contour)
        if cnt_area > area:
            x, y, w, h = get_rect_around_contour(contour)
            hand_contour = np.array(contour)
            max_contour_dict[cnt_area] = (np.array([x, y, w, h]), hand_contour)
    if len(max_contour_dict.keys()) > 0:
        max_area = max(max_contour_dict.keys())
        return max_contour_dict[max_area]

    return np.array([x, y, w, h]), hand_contour


def get_rect_around_contour(contour):
    """
    Draws a bounding box given a contour
    :param image: input image to draw bouding box
    :param contour: contour data points
    :return: (x,y,w,h) of the box drawn
    """
    assert isinstance(contour, np.ndarray), 'contour should be a numpy array'
    (x, y, w, h) = cv2.boundingRect(contour)
    return x, y, w + x, h + y


def get_filtered_ioi(ioi_candidates, point):
    assert isinstance(ioi_candidates, list)
    assert isinstance(point, tuple)

    for ioi_candidate in ioi_candidates:
        if cv2.pointPolygonTest(ioi_candidate, point, False) > 0:
            return ioi_candidate
    return None


def fill_ioi(image, contour):
    assert isinstance(image, np.ndarray)
    assert isinstance(contour, np.ndarray)
    mask = np.zeros_like(image)
    if contour is not None:
        cv2.drawContours(mask, [contour], -1, (1, 1, 1), -1)

    return mask


def get_median_depth_contour(depth_frame):
    """
    Calcuates the median depth associated with hand boundingbox
    :param box: np.ndarry bounding box 
    :praram depth_frame: numpy.ndarray depth image
    :return: median depth associated with hand boundingbox
    """
    assert isinstance(box, np.ndarray)
    assert isinstance(depth_frame, np.ndarray)

    x1, y1, x2, y2 = box
    depth_crop = depth_frame[y1:y2, x1:x2]

    depth_crop = depth_crop[depth_crop > 0]
    median_depth = np.median(depth_crop.reshape(-1))
    print("median depth", median_depth)
    return median_depth


def get_median_depth(mask, depth_map):
    """
    Calcuates the median depth associated with hand boundingbox
    :param box: np.ndarry bounding box 
    :praram depth_frame: numpy.ndarray depth image
    :return: median depth associated with hand boundingbox
    """
    assert isinstance(mask, np.ndarray)
    assert isinstance(depth_map, np.ndarray)
    assert len(depth_map.shape) == 2
    white_areas = np.where(mask == 1)
    median_mask = depth_map[white_areas[0], white_areas[1]]  # =125
    median_mask = median_mask[median_mask > 0]
    median_depth = np.median(median_mask)

    return median_depth


def get_bbox_center(box):
    assert isinstance(box, np.ndarray)
    assert box.shape == (4,)
    return int((box[2] + box[0]) / 2), int((box[1] + box[3]) / 2)


def get_mask_ioi(depth_map, mask, thres=50, hand_contour=None):
    """
    Create a segmentation mask for item within the box
    :param image: np.ndarray rgb image
    :param depth_map: np.ndarray containing single channel depth map
    :param box: np.ndarray of bounding coordinates (x1,y1,x2,y2)
    :return depth_mask: mask 
    """

    assert isinstance(depth_map, np.ndarray)
    # assert isinstance(box, np.ndarray)

    median_depth = get_median_depth(mask, depth_map)

    mask_image = depth_map.copy()
    if math.isnan(median_depth):
        return np.zeros_like(mask_image)

    mask_image[mask_image > (median_depth + thres)] = 0
    mask_image[mask_image < (median_depth - thres)] = 0
    mask_image[mask_image != 0] = 1
    if hand_contour is not None:
        hand_contour = hand_contour.reshape(-1, 2)
        y_min = np.min(hand_contour[:, 1])
        mask_image[:y_min, :] = 0
    return mask_image


def get_depth_pipeline():
    """
    Creates and configures pipeline for Depth streaming
    :return: pipeline and align from which frames can be extracted
    """
    # VideoCapture
    print('Creating Pipeline...')
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    print('Configuring Pipeline...')
    config = rs.config()

    config.enable_stream(rs.stream.depth, DEPTH_IMAGE_WIDTH,
                         DEPTH_IMAGE_HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, RGB_IMAGE_WIDTH,
                         RGB_IMAGE_HEIGHT, rs.format.bgr8, FPS)
    # Start streaming
    print('Starting Stream...')
    profile = pipeline.start(config)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: ", depth_scale)
    align_to = rs.stream.color
    align = rs.align(align_to)

    align_to = rs.stream.color
    align = rs.align(align_to)
    return pipeline, align


def get_aligned_frames(pipeline, align):
    """
    Get depth and color frames to after alignment
    :param pipeline: pipeline object of pyrealsense
    :param align: align object of pyrealsense
    :return: color frame and aligned depth frame
    """
    frames = pipeline.wait_for_frames()

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    return color_frame, aligned_depth_frame


def binarize(seg_mask, thres=0.5):
    seg_mask[seg_mask > thres] = 1
    seg_mask[seg_mask <= thres] = 0
    return seg_mask


def convert_to_numpy(color_frame, aligned_depth_frame):
    """
    Converts frame returned by depth camera to numpy
    :param color_frame: color frame to cbe converted to numpy
    :param aligned_depth_frame: depth frame to be converted to numpy
    :return: numpy color frame and depth 
    """
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    frame = np.asanyarray(color_frame.get_data())
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, depth_image


def to_PIL(frame):
    """
    Converts input image to PIL
    :param frame: Input frame to be converted to PIL
    :return: PIL Image
    """
    assert isinstance(frame, np.ndarray)
    image = Image.fromarray(frame)
    image = image.convert('RGB')
    return image


def segment_hand(image, transform, seg_model):
    """
    Segments the hand given RGB Image and segmentation model
    :param image: Input image to segment hand
    :param transform: pytorch transforms to apply to the image
    :param seg_model: segmentation model
    :return: binary segmentation mask
    """
    assert isinstance(image, np.ndarray)
    img_height, img_width, _ = image.shape
    image = to_PIL(image)
    # Image to feed into the model
    image = transform(image)
    image_tensor = torch.unsqueeze(image, 0).cuda()

    # Get binary mask from the image
    seg_mask = seg_model(image_tensor)
    seg_mask = seg_mask.cpu().detach().numpy().squeeze(0)

    seg_mask = cv2.resize(seg_mask, (img_width, img_height))
    seg_mask[seg_mask > 0] = 1
    seg_mask[seg_mask < 0] = 0

    return seg_mask


def mask_frame(image, mask):
    """
    Applies mask to the image
    :param image: image to apply mask to
    :param mask: mask to apply to image
    :return: image containing the segmentated hand
    """
    # For segmentation mask display

    mask[mask == 1] = 2
    mask[mask == 0] = 1

    # segmentation will display final output
    segmentation = image.copy()
    segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB).astype(float)

    segmentation[:, :, 0] = segmentation[:, :, 0] * mask
    segmentation[:, :, 1] = segmentation[:, :, 1] * mask
    segmentation[segmentation > 255] = 255
    return segmentation.astype("uint8")


def width_points(mask, center_x, center_y):
    """
    :param mask: binary mask to track values
    :param array: Input array
    :param point: row_point
    """
    assert isinstance(mask, np.ndarray)
    assert isinstance(center_x, int)
    assert isinstance(center_y, int)
    mask = mask.copy()
    left_black_max, right_black_min = None, None

    if center_x != 0 and center_y != 0:
        if center_y >= mask.shape[0] - 1:
            center_y = mask.shape[0] - 1
        mask = mask.copy()
        black_points_left = np.where(mask[center_y, :center_x] == 0)
        black_points_right = np.where(mask[center_y, center_x:] == 0)
        if black_points_left[0].size != 0 and black_points_right[0].size != 0:
            left_black_max = np.max(black_points_left[0])
            right_black_min = np.min(black_points_right[0]) + center_x

    return left_black_max, right_black_min


def get_multiple_widths(mask, center_x, center_y, height, number_of_points=10, gradient_thres=50):
    if height < number_of_points:
        #print("ERROR: height < number of points")
        return mask.shape[0] - 1
    jump_size = int(height / number_of_points)
    prev_width = 0
    for i in range(number_of_points):
        left, right = width_points(mask, center_x, center_y + (jump_size * i))
        if left is None or right is None:
            return mask.shape[0] - 1
        width = abs(left - right)
        if ((width - prev_width) > gradient_thres and prev_width != 0):
            return center_y + (jump_size * i) + 50
        prev_width = width
    return mask.shape[0] - 1
