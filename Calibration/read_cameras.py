import pyrealsense2 as rs
import numpy as np
import cv2
import time


class Realsense:

    def __init__(self,camera_name, depth_res, rgb_res, fps):
        self.depth_res = depth_res
        self.rgb_res = rgb_res
        self.fps = fps
        self.cam_id = None
        self.camera_name=camera_name
        self.init_pipeline()

    def init_pipeline(self):
        try:
             
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.camera_name)
            config.enable_stream(
                rs.stream.depth, self.depth_res[0], self.depth_res[1], rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.rgb_res[0],
                                 self.rgb_res[1], rs.format.bgr8, self.fps)
            profile = self.pipeline.start(config)

            align_to = rs.stream.color
            self.align = rs.align(align_to)
             
        except Exception as e:
            print (e)

    def get_frames(self):
        """
        Get depth and color frames to after alignment
        :param pipeline: pipeline object of pyrealsense
        :param align: align object of pyrealsense
        :return: color frame and aligned depth frame
        """
        while True:
            frames = self.pipeline.wait_for_frames()

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue
            color_frame = np.asarray(color_frame.get_data())
            depth_frame = np.asarray(aligned_depth_frame.get_data())

            yield (color_frame, depth_frame)

    def put_text(self, color_frame, bottom_left, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7
        fontColor = (255, 255, 255)
        lineType = 2
        cv2.putText(color_frame, text,
                    bottom_left,
                    font,
                    fontScale,
                    fontColor,
                    lineType)

    def assign_cam_id(self):
        cam_mapping = {1: "Near Right", 2: "Near Left",
                       3: "Far Left", 4: "Far Right"}
        while True:
            color_frame, _ = next(self.get_frames())
            self.put_text(color_frame, (int(color_frame.shape[1]/1.7), int(color_frame.shape[0]/4)),
                          "Press 1 for near right cam")
            self.put_text(color_frame, (int(color_frame.shape[1]/1.7),int( color_frame.shape[0]/4) + 30),
                          "Press 2 for near left cam")
            self.put_text(color_frame, (int(color_frame.shape[1]/1.7), int(color_frame.shape[0]/4) + 60),
                          "Press 3 for far left cam")
            self.put_text(color_frame, (int(color_frame.shape[1]/1.7), int(color_frame.shape[0]/4) + 90),
                          "Press 4 for far right cam")

            cv2.imshow("", color_frame)
            key = cv2.waitKey(1)
            assigned = False
            for i in range(1, 5):
                if key == ord(str(i)):
                    self.cam_id = i
                    assigned = True
                    break
            if assigned:
                break

        self.put_text(color_frame, (int(
            color_frame.shape[1]/2.3), int(color_frame.shape[0]/4)+200), cam_mapping[self.cam_id]+" Assigned")
        cv2.imshow("", color_frame)
        cv2.waitKey()

        cv2.destroyAllWindows()
