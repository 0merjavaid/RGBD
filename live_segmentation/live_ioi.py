from utils import *

# Setting GPU device
torch.cuda.set_device(0)
torch.backends.cudnn.benchmark = True

# Hyperparameters
batch_size = 1
img_size = 256
ioi_width_thres=50
weights_path = 'weights/UNET_5.pt'


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
    i =0
    while ret: 
        i+=1 
     
        print (i)
        color_frame, aligned_depth_frame = get_aligned_frames(pipeline, align)
        if not aligned_depth_frame or not color_frame:
            continue
        #continue 

        frame, depth_image = convert_to_numpy(color_frame, aligned_depth_frame)
        original_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        masked_time=time.time()
        seg_mask = segment_hand(frame, transform, seg_model) #~15 milliseconds for 256
        
        mask = seg_mask.copy() 
         
        contours = get_contours(seg_mask) # ~1 millisecond 
        
        bounding_box, hand_contour = refine_contours(contours) #~0.1 millisecond
        
        overlay_mask, ioi_mask = np.ones_like(original_image),np.ones_like(original_image)
        
        if hand_contour is not None:
            print ("hand con",len(hand_contour))
            mask_ioi = get_mask_ioi(depth_image, mask, 90, hand_contour)
            
            #cv2.imshow("mask image",(mask_ioi).astype("uint8"))
            
            
            ioi_candidates = get_contours(mask_ioi*255)
            center_x, center_y = get_bbox_center(bounding_box)
            ioi = get_filtered_ioi(ioi_candidates, (center_x, center_y))

            if ioi is None:
                continue
            ioi=ioi.reshape(-1,2)
            ioi_mask = fill_ioi(frame, ioi)
            cv2.imshow("image",(ioi_mask*255).astype("uint8"))
            hand_bottom_y=int(np.max(hand_contour.reshape(-1,2)[:,1]))
            ioi_height=np.max(ioi[:,1])-hand_bottom_y
            ioi_mask_before=ioi_mask.copy()
            left_black_max,right_black_min=width_points(ioi_mask[:,:,0],center_x,hand_bottom_y) 
            if left_black_max is not None and right_black_min is not None:
                ioi_mask[:,:(left_black_max-ioi_width_thres)] = 0
                ioi_mask[:,(right_black_min+ioi_width_thres):] = 0
               
            
            y_threshold=get_multiple_widths(ioi_mask[:,:,0],center_x,hand_bottom_y,ioi_height)      
            ioi_mask[y_threshold:,:]=0  



            overlay_mask = ioi_mask.copy()
        
            overlay_mask[overlay_mask == 1] = 3
            overlay_mask[overlay_mask == 0] = 1
        print (time.time()-masked_time)
        overlay_image = original_image.copy().astype(float)*overlay_mask
        overlay_image[overlay_image > 255] = 255
        overlay_image[:, :, :2] = original_image[:, :, :2].copy()
        two_way_show = np.concatenate(
        (ove
            rlay_image, original_image*i
            oi_mask), axis=1)
        cv2.imshow("ismage",two_way_show.astype("uint8"))
       
        key = cv2.waitKey(1)
        if key == ord('q'):
             break
         
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()