import torch
def get_iou(tracker_predicted_bbox, ioi_bbox):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    tracker_predicted_bbox : np.array
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    ioi_bbox : np.array
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    t_x1, t_y1, t_x2, t_y2 = tracker_predicted_bbox.astype(float)
    ioi_x1, ioi_y1, ioi_x2, ioi_y2 = ioi_bbox.astype(float)
    assert t_x1 <= t_x2
    assert t_y1 <= t_y2
    assert ioi_x1 < ioi_x2
    assert ioi_y1 < ioi_y2

    # Determine coordinates of the intersection rectangle
    x_left = max(t_x1, ioi_x1)
    y_top = max(t_y1, ioi_y1)
    x_right = min(t_x2, ioi_x2)
    y_bottom = min(t_y2, ioi_y2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = int(
        abs((x_right - x_left)) * abs((y_bottom - y_top))
    )

    # compute the area of both AABBs
    tracker_predicted_bbox_area = (t_x2 - t_x1) * (t_y2 - t_y1)
    ioi_bbox_area = (ioi_x2 - ioi_x1) * (ioi_y2 - ioi_y1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    divider = float(int(tracker_predicted_bbox_area) + int(ioi_bbox_area) -
                    int(intersection_area))
    iou = intersection_area / divider

    assert iou >= 0.0
    assert iou <= 1.0, str([tracker_predicted_bbox, ioi_bbox])
    return iou


def decode_rgb_bytes(rgb_data_bytes):
    rgb_data = cv2.imdecode(np.fromstring(rgb_data_bytes, dtype=np.uint8), 1)
    return rgb_data


def parse_model_config(path):
    """Parse the yolo-v3 layer configuration file

    returns module definitions
    """
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip()
             for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs


def parse_data_config(path):
    """Parse the data configuration file"""
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


def load_classes(path):
    """
    Load class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Return the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,
                                          0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,
                                          0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    """Non Maximal Suppression

    Remove detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_mask = (image_pred[:, 4] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5:5 + num_classes], 1, keepdim=True)
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf,
        # class_pred)
        detections = torch.cat(
            (image_pred[:, :5], class_conf.float(), class_pred.float()), 1)
        # Iterate through all predicted classes
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # Get the detections with the particular class
            detections_class = detections[detections[:, -1] == c]
            # Sort the detections by maximum objectness confidence
            _, conf_sort_index = torch.sort(
                detections_class[:, 4], descending=True)
            detections_class = detections_class[conf_sort_index]
            # Perform non-maximum suppression
            max_detections = []
            while detections_class.size(0):
                # Get detection with highest confidence and save as max
                # detection
                max_detections.append(detections_class[0].unsqueeze(0))
                # Stop if we're at the last detection
                if len(detections_class) == 1:
                    break
                # Get the IOUs for all boxes with lower confidence
                ious = bbox_iou(max_detections[-1], detections_class[1:])
                # Remove detections with IoU >= NMS threshold
                detections_class = detections_class[1:][ious < nms_thres]

            max_detections = torch.cat(max_detections).data
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output


def build_targets(pred_boxes, target, anchors, num_anchors, num_classes, dim, ignore_thres, img_dim):
    nB = target.size(0)
    nA = num_anchors
    nC = num_classes
    dim = dim
    mask = torch.zeros(nB, nA, dim, dim)
    tx = torch.zeros(nB, nA, dim, dim)
    ty = torch.zeros(nB, nA, dim, dim)
    tw = torch.zeros(nB, nA, dim, dim)
    th = torch.zeros(nB, nA, dim, dim)
    tconf = torch.zeros(nB, nA, dim, dim)
    tcls = torch.zeros(nB, nA, dim, dim, num_classes)

    nGT = 0
    nCorrect = 0
    for b in range(nB):
        for t in range(target.shape[1]):
            if target[b, t].sum() == 0:
                continue
            nGT = nGT + 1
            # Convert to position relative to box
            gx = target[b, t, 1] * dim
            gy = target[b, t, 2] * dim
            gw = target[b, t, 3] * dim
            gh = target[b, t, 4] * dim
            # Get grid box indices
            gi = int(gx)
            gj = int(gy)
            # Get shape of gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # Get shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate(
                (np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # Calculate iou between gt and anchor shape
            anch_ious = bbox_iou(gt_box, anchor_shapes)
            # Find the best matching anchor box
            best_n = np.argmax(anch_ious)
            best_iou = anch_ious[best_n]
            # Get the ground truth box and corresponding best prediction
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            pred_box = pred_boxes[b, best_n, gj, gi].unsqueeze(0)

            # Masks
            mask[b, best_n, gj, gi] = 1
            # Coordinates
            tx[b, best_n, gj, gi] = gx - gi
            ty[b, best_n, gj, gi] = gy - gj
            # Width and height
            tw[b, best_n, gj, gi] = math.log(gw / anchors[best_n][0] + 1e-16)
            th[b, best_n, gj, gi] = math.log(gh / anchors[best_n][1] + 1e-16)
            # One-hot encoding of label
            tcls[b, best_n, gj, gi, int(target[b, t, 0])] = 1
            # Calculate iou between ground truth and best matching prediction
            iou = bbox_iou(gt_box, pred_box, x1y1x2y2=False)
            tconf[b, best_n, gj, gi] = 1

            if iou > 0.5:
                nCorrect = nCorrect + 1

    return nGT, nCorrect, mask, tx, ty, tw, th, tconf, tcls


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.from_numpy(np.eye(num_classes, dtype='uint8')[y])


def image_sanity_check(image):
    # Sanity check input image dimensions and type
    assert type(image) == np.ndarray
    assert image.ndim == 3
    height, width, channels = image.shape
    assert height < 1280
    assert width < 1920
    assert channels == 3


def blur_roi(image, boxes):
    blur = image.copy()
    blur = cv2.GaussianBlur(blur, (125, 125), 0)
    for box in boxes:
        blur[box[1]:box[3], box[0]:box[2]] = image[
            box[1]:box[3], box[0]:box[2]]
    return blur


def bbox_sanity_check(bbox):
    assert type(bbox) == np.ndarray, str(type(bbox))
    assert bbox.shape == (1, 4) or bbox.shape == (4, ), str(bbox.shape)
    x1, y1, x2, y2 = bbox
    assert x1 < x2, str(bbox)
    assert y1 < y2, str(bbox)


def make_img_grid_4x4(im_array):
    assert im_array is not None
    assert type(im_array) is list
    assert len(im_array) == 16 or len(im_array) == 4

    col1_im = cv2.hconcat(im_array[0:4])

    if len(im_array) == 16:
        col2_im = cv2.hconcat(im_array[4:8])
        col3_im = cv2.hconcat(im_array[8:12])
        col4_im = cv2.hconcat(im_array[12:])
        final_im = cv2.vconcat((col1_im, col2_im, col3_im, col4_im))
    else:
        final_im = col1_im

    return final_im

