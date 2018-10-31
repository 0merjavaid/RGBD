# Import yolo for object detection
import yolo
from depth_utils import *

threshold = 150

# Setup depth stream
pipeline, config = setup_pipeline()
align = align_frames(pipeline, config)
color_frame, aligned_depth_frame = get_aligned_frames(pipeline, align)
color_image, depth_image = rgbd_to_numpy(color_frame, aligned_depth_frame)

# Init YOLO model
YOLO = yolo.YOLO()

try:
    while True:
        # Get frameset of color and depth
        color_frame, aligned_depth_frame = get_aligned_frames(pipeline, align)
        
        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        color_image, depth_image = rgbd_to_numpy(color_frame, aligned_depth_frame)
        color_image_ioi = color_image.copy()
        
        # Apply YOLO to this image and get bounding boxes
        hands, items= YOLO.get_item_of_interest(color_image)
        result = run(hands,items)
        
        # get hand centroid
        hand_centroids = []
        if len(hands) > 0 and result.shape[1] != 0:         
            hand_of_interest = get_hand_of_interest(hands, result)
            if hand_of_interest is not None:
                median_depth_hand = get_median_depth(hand_of_interest, depth_image)
                hand_centroids.append((median_depth_hand, hand_of_interest))
                color_image = draw_boundingBox(color_image, hand_of_interest, str('Depth: ' + str(median_depth_hand)))
        
        # Get Item centroid
        item_centroids = []    
        for box in items:
            if box is not None:
                median_depth_item = get_median_depth(box, depth_image)
                item_centroids.append((median_depth_item, box))
                color_image = draw_boundingBox(color_image, box, str('Depth: ' + str(median_depth_item)), box_color=(255,0,0))
        

        # All hands and items are detected
        if (len(hand_centroids) != 0) and (len(item_centroids) != 0):
            final_box = get_item_of_interest(hand_centroids, item_centroids, threshold)
            if final_box is not None:
                color_image_ioi = draw_boundingBox(color_image_ioi, final_box, box_color=(0,0,0), box_thickness=4)              

        show_image('yolo', color_image)
        show_image('ioi', color_image_ioi)
finally:
    pipeline.stop()