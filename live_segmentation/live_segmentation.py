from utils import *

# Setting GPU device
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

# Hyperparameters
batch_size = 1
img_size = 512

weights_path = 'weights/UNET_2.pt'


def main():
    # Get transforms
    print('Loading transforms...')
    transform = get_transforms(img_size)

    # Get model
    print('Loading model...')
    seg_model = get_segmentation_model(weights_path)
    seg_model.eval()

    pipeline, align = get_depth_pipeline()

    ret = True
    while ret:
        color_frame, aligned_depth_frame = get_aligned_frames(pipeline, align)
        if not aligned_depth_frame or not color_frame:
            continue
        frame, depth_image = convert_to_numpy(color_frame, aligned_depth_frame)
        original_image = frame.copy()
        seg_mask = segment_hand(frame, transform, seg_model)
        contours = get_contours(seg_mask)
        masked_frame = mask_frame(original_image, seg_mask)
        draw_contours(masked_frame, contours)

        # Display frame
        cv2.imshow('window', masked_frame.astype("uint8"))
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
