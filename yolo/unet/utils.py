from fastai.conv_learner import *
from fastai.dataset import *
import cv2
import numpy as np
from torchvision import models, transforms

import pyrealsense2 as rs
from unet.unet import Unet34

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
FPS = 30


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
    m = to_gpu(Unet34(m_base))
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


def draw_contours(image, contours):
    """
    Draw bounding box around contour
    :param image: input image to draw boxes on
    :param contours: contours around which bounding box will be drawn
    :return: (X,y,w,h) of the last contour
    """
    assert isinstance(image, np.ndarray), 'Image should be a numpy array'
    x, y, w, h = 0, 0, 0, 0
    for contour in contours:
        area = get_contour_area(contour)
        if area > 10000:
            x, y, w, h = draw_rect_around_contour(image, contour)
    return x, y, w+x, h+y


def draw_rect_around_contour(image, contour):
    """
    Draws a bounding box given a contour
    :param image: input image to draw bouding box
    :param contour: contour data points
    :return: (x,y,w,h) of the box drawn
    """
    assert isinstance(contour, np.ndarray), 'contour should be a numpy array'
    assert isinstance(image, np.ndarray), 'Image should be a numpy array'
    assert len(image.shape) == 3, f'Image should have three channels, got {len(image.shape)} channels'

    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    return x, y, w, h


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

    config.enable_stream(rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.bgr8, FPS)
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
    print('frames')

    # Align the depth frame to color frame
    aligned_frames = align.process(frames)

    # Get aligned frames
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    return color_frame, aligned_depth_frame


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
    img_height, img_width, _ = image.shape
    
    image = to_PIL(image)
    # Image to feed into the model
    image = transform(image)
    image_tensor = torch.unsqueeze(image, 0).cuda()

    # Get binary mask from the image
    seg_mask = seg_model(image_tensor)
    seg_mask = seg_mask.cpu().detach().numpy().squeeze(0)
    seg_mask[seg_mask > 0] = 1
    seg_mask[seg_mask < 0] = 0
    
    # Resize to captured image
    seg_mask = cv2.resize(seg_mask, (img_width, img_height)) 
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
    segmentation = image
    segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB)

    segmentation[:, :, 0] = segmentation[:, :, 0] * mask
    segmentation[:, :, 1] = segmentation[:, :, 1] * mask
    return segmentation
