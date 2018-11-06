from utils import *

# Setting GPU device
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

# Hyperparameters
batch_size = 1
img_size = 512

weights_path = 'weights/UNET_mini.pt'


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
    i=0
    while ret: 
        i+=1
        #print ("frame ," ,i)
        frame_start_time = time.time()
        color_frame, aligned_depth_frame = get_aligned_frames(pipeline, align)
        if not aligned_depth_frame or not color_frame:
            continue
        #continue
        start_time = time.time()
        frame, depth_image = convert_to_numpy(color_frame, aligned_depth_frame)
        original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hand_start_time = time.time()
        seg_mask = segment_hand(frame, transform, seg_model)
        # if i==100:
        #     fps=time.time()
        # if i>100:
        #     if time.time()-fps>=10:
        #         print (i)
        #         break
        seg_mask[seg_mask>0.5]=1
        seg_mask[seg_mask<=0.5]=0
        mask = seg_mask.copy() 
        hand_end_time = time.time()
        contours = get_contours(seg_mask)
        
        
       
        masked_frame = mask_frame(original_image, seg_mask)
        # cv2.imshow('hath', masked_frame)
        # cv2.waitKey(1)
        # continue
        hath = masked_frame.copy()
        bounding_box, hand_contour = draw_contours(masked_frame, contours)

        
        mask_ioi = get_mask_ioi(depth_image, mask, 90, hand_contour)
        
        #cv2.imshow("mask image",(mask_ioi).astype("uint8"))
        
        masked_frame[:, :, 0] *= mask_ioi
        masked_frame[:, :, 1] *= mask_ioi
        masked_frame[:, :, 2] *= mask_ioi
        ioi_candidates = get_contours(mask_ioi*255)
        center_x, center_y = int(
            (bounding_box[2]+bounding_box[0])/2), int((bounding_box[1]+bounding_box[3])/2)
        ioi = get_filtered_ioi(ioi_candidates, (center_x, center_y))
        ioi_mask = draw_ioi(frame, ioi)
        ioi_end_time = time.time()
                        
        overlay_mask = ioi_mask.copy()
        overlay_mask[overlay_mask == 1] = 3
        overlay_mask[overlay_mask == 0] = 1
        overlay_image = original_image.copy().astype(float)*overlay_mask
        overlay_image[overlay_image > 255] = 255
        overlay_image[:, :, :2] = original_image[:, :, :2].copy()
        two_way_show = np.concatenate(
            (overlay_image, original_image*ioi_mask), axis=1)

        cv2.imshow("image",two_way_show.astype("uint8"))
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        #print ("total time per frame: ", time.time()-frame_start_time, "  hand time: ",
        #                               hand_end_time-hand_start_time, " IoI time: ",
        #                               ioi_end_time-hand_end_time, "render time: ",
        #                               time.time()-ioi_end_time, " Frame get time: ",
        #                               start_time-frame_start_time, "\n\n")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()