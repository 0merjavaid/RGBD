import os
import glob
import shutil
import matplotlib.pyplot as plt
#from extract_crops import save_crops
import pyrealsense2 as rs
import numpy as np
import cv2
import os
import argparse


def get_pipeline(directory, rgb_res=(848, 480), depth_res=(640, 360), fps=30):
    config = rs.config()

    rs.config.enable_device_from_file(config, directory, False)
    pipeline = rs.pipeline()

    config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, fps)

    config.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, fps)
    align_to = rs.stream.color
    align = rs.align(align_to)
    return pipeline, config, align

def fuck():
    try:
        pipeline,config,align= get_pipeline("/home/haroonrashid/omer/fovea/data/videos/cam1.bag", rgb_res=(848, 480), depth_res=(640, 360), fps=30)
    
        pipeline.start(config)
    except Exception as e:
        print (e)
        return


    while True:
        try:
            frames = pipeline.wait_for_frames()
        except Exception as e:
            print (e)
            print ("video completed")
            return
        aligned_frames = align.process(frames) 

        depth_frame = aligned_frames.get_depth_frame()

        depth_image = np.asanyarray(depth_frame.get_data())

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imshow("",color_image)
        cv2.waitKey(1)
    cv2.destroyAllWindows()


def process_bag(path, barcode, crop_save_dir, rgb_res, depth_res):
    assert isinstance(rgb_res, tuple)
    assert isinstance(depth_res, tuple)
    assert os.path.exists(path)
    try:
        pipeline, config, align = get_pipeline(
            path, rgb_res=rgb_res, depth_res=depth_res)
        pipeline.start(config)
    except Exception as e:
        print (e)
        return

    frame_no = 0
    while True:
        try:
            frames = pipeline.wait_for_frames()
        except Exception as e:
            print (e)
            print ("video completed")
            return
        aligned_frames = align.process(frames)
        frame_no += 1

        depth_frame = aligned_frames.get_depth_frame()

        depth_image = np.asanyarray(depth_frame.get_data())

        color_frame = aligned_frames.get_color_frame()
        color_image = np.asanyarray(color_frame.get_data())
        box, hand_box = (save_crops(color_image, depth_image, "unet"))
        dirs = os.path.join(crop_save_dir, barcode)
        if box is not None:
            os.makedirs(dirs, exist_ok=True)
            file = os.path.join(os.path.join(
                crop_save_dir, barcode), (str(frame_no)+".jpg"))
            color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            crop = color_image[box[1]:box[3], box[0]:box[2]]
            cv2.imwrite(file, crop)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", help="root directory where all folders containing bag files are present")

    parser.add_argument("--rgb_res", default=(
        848, 480), help="resolution of rgb", nargs='+', type=int)
    parser.add_argument("--depth_res", default=(
        640, 360), help="resolution of depth example --depth_res 640 360", nargs='+', type=int)

    args = vars(parser.parse_args())

    #root_dir = "/home/haroonrashid/Zain/Cameras/librealsense/python/Data/version1/tracking"
    root_dir = args["root_dir"]
    rgb_res = tuple(args["rgb_res"])
    depth_res = tuple(args["depth_res"])
    i = 0
    folders = os.listdir(root_dir)
    for folder in folders:

        barcode = folder.split("-")[0]
        folder = os.path.join(root_dir, folder)
        files = (os.listdir(folder))
        for file in files:

            file_path = os.path.join(folder, file)
            file_size = (os.path.getsize(file_path))

            if file_size > 10000000 and "bag" in file_path:
                crop_save_dir = file_path.replace(".bag", "_crops")
                os.makedirs(crop_save_dir, exist_ok=True)
                print (crop_save_dir)
                process_bag(file_path, barcode,
                            crop_save_dir, rgb_res, depth_res)

if __name__ == "__main__":
    fuck()