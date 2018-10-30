import pyrealsense2 as rs
import numpy as np


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

