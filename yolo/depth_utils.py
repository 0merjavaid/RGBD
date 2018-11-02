import pyrealsense2 as rs
import numpy as np
import cv2
import unet.UNET as unet

#from unet.utils import *


def setup_pipeline(img_dims=(1280, 720), depth_dims=(848, 480), fps=30):
    """
    Setup and configure pipeline with required image dimensions and fps
    :param img_dims: A tuple containing (width, height) of rgb image, default = (1280, 720)
    :param depth_dims: A tuple containing (width, height) of rgbd image, default = (848, 480)
    :param fps: Frames per second to stream, default = 30
    :return: Returns pipeline and config object 
    """
    assert isinstance(img_dims, (tuple, list))
    assert isinstance(depth_dims, (tuple, list))
    assert isinstance(fps, (int,float))
    assert len(img_dims) == 2
    assert len(depth_dims) == 2

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    config = rs.config()

    # Get required resolution for color and depth
    img_width, img_height = img_dims
    depth_width, depth_height = depth_dims

    # Setup config with desired fps and resolution
    config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, img_width, img_height, rs.format.bgr8, fps)

    return pipeline, config


def align_frames(pipeline, config, clip_dist = 1):
    """
    Alignment of color and depth images
    :param pipeline: Pipeline object of pyrealsense2
    :param config: config object of pyrealsense2
    :clip_dist: Clipping distance in meters, default=1
    :return: align object to align frames
    """
    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = clip_dist #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)
    return align


def get_aligned_frames(pipeline, align):
    """
    Get aligned color and depth frames
    :param pipeline: Pipeline object of pyrealsense2
    :param align: pyrealsense2 align object
    :return: aligned color and depth frames
    """
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame() 
    
    color_frame = aligned_frames.get_color_frame()

    return color_frame, aligned_depth_frame


def rgbd_to_numpy(color_frame, depth_frame):
    """
    Convert pyreaslsense video frame and depth frame to numpy
    :param color_frame: pyrealsense2 video frame
    :param depth_frame: pyrealsense2 depth frame
    :return: color image and depth image as numpy array
    """
    assert isinstance(color_frame, rs.pyrealsense2.video_frame)
    assert isinstance(depth_frame, rs.pyrealsense2.depth_frame)

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return color_image, depth_image


def run(bboxes1, bboxes2):
    """
    Calcualtes Intersection over Union (IOU) of the two list of bounding boxes
    :param bboxes1: List containing Bounding Boxes 1
    :param bboxes2: List containing BOunding Boxes 2
    :return: IOU of all the bboxes1 with bboxes2
    """
    x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
    boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
    boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
    iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
    return iou


def get_hand_of_interest(hands, result):
    """
    Gives hand that has max overlap with the item 
    :param hands: numpy.ndarray of bounding boxes containing hands
    :param result: np.ndarray containing iou of all hands with items
    :return: Hand that has max overlap with any of the items
    """
    assert isinstance(hands, np.ndarray)
    assert isinstance(result, np.ndarray)

    return hands[np.unravel_index(result.argmax(), result.shape)[0]]
    

def get_median_depth(box, depth_frame):
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
    median_depth = np.median(depth_crop.reshape(-1))
    return median_depth
    

def draw_boundingBox(image, box, text='', box_color=(0,255,0), text_color= (0,0,0), box_thickness=2):
    """
    Draw a bounding box around on image
    :param image: np.ndarray to draw bounding box
    :param box: np.ndarray of bounding coordinates (x1,y1,x2,y2)
    :param text: text to put on bounding box, default=''
    :param box_color: tuple contating rgb color for bounding box, default = (0,255,0)
    :param text_color: tuple containing rgb color for text, default = (0,0,0)
    :param thickness: thickness of bounding box, default = 2
    :return: np.ndarray containing the iou
    """
    assert isinstance(image, np.ndarray)
    assert isinstance(box, np.ndarray)
    assert box.shape == (4,)
    assert len(box_color) == 3
    assert len(text_color) == 3
    assert isinstance(box_thickness, int)
    assert box_thickness > 0

    cv2.putText(image,str(text), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 
               1,text_color,2)
    cv2.rectangle(image,(box[0],box[1]),(box[2],box[3]),box_color,box_thickness)
    return image

def get_depth_diffs(hand, items_list):
    """
    Calculates difference in depth between hand and each item
    :param hand: tuple containing (hand_depth, hand_box)
    :param items_list: List of tuples (item_depth, item_box)
    :return diffs: list containg diff in depth between hand with items
    """
    assert isinstance(hand, tuple)
    assert isinstance(items_list, list)

    diffs = []
    hand_depth = hand[0]
    hand_box = hand[1]
    # Iterate over a;; items to find distance with hand
    for item in items_list:
        item_depth = item[0]
        item_box = item[1]

        # Calculate IOU of hand with item
        iou = run(np.array([hand_box]), np.array([item_box]))

        if iou > 0:
            depth_diff = abs(hand_depth - item_depth)
            diffs.append((item_box, depth_diff))
    return diffs



def get_item_of_interest(hand_list, items_list, threshold=50):
    """
    Extracts Item of Interest given hand and multiple items
    :param hand_list: List of tuples (median_depth, box)
    :param items_list: List of tuples (median_depth, box)
    :param threshold: depth threshold, default = 150 
    :return final_box: final_box containing the item_of_interest
    """
    assert isinstance(hand_list, list)
    assert isinstance(items_list, list)

    # assuming only one hand at index 0
    hand = hand_list[0]

    hand_depth = hand[0]
    hand_box = hand[1]

    diffs = get_depth_diffs(hand, items_list)
    
    item_boxes = sorted(diffs, key=lambda x: x[1])

    final_box = None
    if len(item_boxes) > 0:
        final_tuple = item_boxes[0]
        final_depth = final_tuple[1]
        if final_depth < threshold:
            final_box = final_tuple[0]

    return final_box

def show_image(window_name, image):
    """
    Displays image in a window
    :param window_name: window name as string
    :param image: np.ndarray image
    """

    assert isinstance(window_name, str)
    assert isinstance(image, np.ndarray)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, image)
    cv2.waitKey(1)


def segment_hands(image, area_thres=1000):
    """
    Retuns list of hands bounding boxes and segmentation mask
    :param image: np.ndarray image
    :param area_thres: Area threshold to consider bounding box a hand 
    """
    seg_model = unet.UNET()
    hands, seg_mask = seg_model.get_hands(image, area_thres)
    return hands, seg_mask


def mask_ioi(depth_map, box, thres = 50):
    """
    Create a segmentation mask for item within the box
    :param image: np.ndarray rgb image
    :param depth_map: np.ndarray containing single channel depth map
    :param box: np.ndarray of bounding coordinates (x1,y1,x2,y2)
    :return depth_mask: mask 
    """
    
    assert isinstance(depth_map, np.ndarray)
    assert isinstance(box, np.ndarray)

    x1,y1,x2,y2 = box
    median_depth = get_median_depth(box, depth_map)
    
    mask = np.zeros_like(depth_map)
    mask[y1:y2, x1:x2] = depth_map[y1:y2, x1:x2]

    mask[mask > (median_depth + threshold)] = 0
    mask[mask < (median_depth - threshold)] = 0

    mask[mask !=0 ] = 1

    return mask



