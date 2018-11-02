from unet.utils import *

class UNET:

    def __init__(self, model_path='unet/weights/UNET_2.pt', img_size =512, batch_size=1):
        """
        Constructor for creating UNET Object
        :param model_path: path to the model weights
        :param img_size: resize to image_size, image_size, default = 512
        :param batch_size: batch_size , default = 1 
        """
        self.device = torch.cuda.set_device(0)
        self.img_size =  img_size
        self.batch_size = batch_size
        self.transforms = get_transforms(img_size)
        self.get_unet(model_path)

    def get_unet(self, model_path):
        """
        Creates model for semantic segmentation
        """
        self.model = get_segmentation_model(model_path)
        self.model.eval()
        

    def get_hands(self, frame, area_thres):
        """
        Segment Hands using UNET model
        :param frame: np.ndarray image
        :param area_thres: Threshold Area of the contour
        """
        assert isinstance(frame, np.ndarray)
        assert isinstance(area_thres, (int, float))
        hands = []
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        seg_mask = segment_hand(frame, self.transforms, self.model)
        contours = get_contours(seg_mask)
        for contour in contours:
            area = get_contour_area(contour)
            if area > area_thres:
                (x,y,w,h) = cv2.boundingRect(contour)
                hands.append((x, y, x+w, y+h))
        return np.array(hands).reshape(-1,4), seg_mask








