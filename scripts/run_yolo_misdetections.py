import os
import glob
import cv2
import argparse
import yolo
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path', help='path to directory containing images to be passed though yolo')
    parser.add_argument(
        '--box_type', default="*txt", help='data type of bounding boxes')

    args = parser.parse_args()
    return args


def get_contours(mask):
    mask[mask > 30] = 255
    mask[mask <= 30] = 0
    contours, _ = cv2.findContours(mask.astype(
        "uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    assert len(contours) > 1
    contours = [i for i in contours if cv2.contourArea(i) > 100]
    return contours


def main():
    total_mds = 0
    total_boxes = 0
    args = get_parser()
    box_paths = glob.glob(os.path.join(args.path, args.box_type))
    for box_path in box_paths:
        assert os.path.exists(box_path)
        mask_path = "/".join(box_path.split("/")[:-1])+"/" +\
            "seg_"+box_path.split("/")[-1].replace("txt", "png")
        assert os.path.exists(mask_path)

        boxes = np.loadtxt(box_path)
        rgb = cv2.imread(box_path.replace("txt", "jpg"))
        im = cv2.imread(mask_path)
        contours = get_contours(im[:, :, 0])

        if len(contours) > 6:
            print (len(contours), "len")

            cv2.imshow("", im)
            k = cv2.waitKey()
            if k == ord("q"):
                break

        del_list = []
        boxes = boxes.reshape(-1, 5)
        total_boxes += boxes.shape[0]
        for box in boxes:
            box = box.astype("int")

            try:
                cv2.rectangle(im, (box[1], box[2]),
                              (box[3], box[4]), (222, 0, 2), 3)
            except:
                print (boxes)
            center = int((box[3]+box[1])/2),     int((box[4]+box[2])/2)

            cv2.circle(im, center, 3, (0, 0, 255), -1)
            for i in range(len(contours)):
                if cv2.pointPolygonTest(contours[i], center, True) > 0:
                    del(contours[i])
                    break
        cv2.drawContours(rgb, contours, -1, (0, 255, 0), 3)

        if (len(contours)) > -10:
            print (len(contours))
            total_mds += (len(contours))
            cv2.imshow("", im)
            cv2.imshow("s", rgb)
            k = cv2.waitKey()
            if k == ord("q"):
                break
    print (total_mds, total_boxes)


if __name__ == "__main__":

    aa = """
    2
    2
    2
    2
    1
    1
    1
    1
    2
    3
    2
    4
    2
    1
    1
    1
    1
    2
    1
    1
    1
    1
    1
    1
    1
    2
    2
    1
    1
    1
    2
    1
    1
    2
    1
    1
    1
    2
    1
    2
    2
    1
    1
    2
    1
    1
    1
    1
    1
    1
    2
    2
    1
    2
    1
    2
    1
    3
    1
    3
    1
    """

    print(sum([int(i) for i in aa.split()]))

    main()
