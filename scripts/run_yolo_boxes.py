import os
import glob
import cv2
import argparse
import yolo
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--images_path', help='path to directory containing images to be passed though yolo')
    parser.add_argument('--save_dir',
                        help='directory to save bounding boxes')
    parser.add_argument('--plot_boxes', default=False,
                        help='True if you want to plol boxes on imags and store in directory')
    parser.add_argument('--type', default="*png",
                        help='type of images')

    args = parser.parse_args()
    return args


def main():
    args = get_parser()
    Infer = yolo.YoloInference()
    assert os.path.exists(args.images_path)
    assert os.path.exists(args.save_dir)

    images = glob.glob(os.path.join(args.images_path, args.type))

    for image in images:
        im = cv2.imread(image)

        boxes = Infer.yolo_localize([im])[0]
        print (image)
        dir = args.save_dir +\
            "_".join(image.split("/")[3:]).replace("png", "txt")
        for box in boxes:
            cv2.rectangle(im, (box[1], box[2]),
                          (box[3], box[4]), (0, 222, 2), 4)
        np.savetxt(dir, boxes)
        cv2.imwrite(dir.replace("txt", "jpg"), im)

if __name__ == "__main__":
    main()
