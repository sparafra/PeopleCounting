from ctypes import *
import random
import os
import cv2
import time
#import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
import Yolo.darknet as darknet

cfg = "/cfg/yolov4.cfg"


class Yolo_V4:
    def __init__(self, ROOT_DIR):

        args = self.parser()
        self.check_arguments_errors(args)
        print(os.path.join(ROOT_DIR, cfg))
        self.network, self.class_names, self.class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )
        # Darknet doesn't accept numpy images.
        # Create one with image we reuse for each detect
        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        self.darknet_image = darknet.make_image(self.width, self.height, 3)
        #input_path = str2int(args.input)

    def get_prediction(self, image):
        frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(self.darknet_image, frame_resized.tobytes())

        detections = darknet.detect_image(self.network, self.class_names, self.darknet_image)


        return detections

    def parser(self):
        parser = argparse.ArgumentParser(description="YOLO Object Detection")
        parser.add_argument("--input", type=str, default=0,
                            help="video source. If empty, uses webcam 0 stream")
        parser.add_argument("--out_filename", type=str, default="",
                            help="inference video name. Not saved if empty")
        parser.add_argument("--weights", default="Yolo/yolov4.weights",
                            help="yolo weights path")
        parser.add_argument("--dont_show", action='store_true',
                            help="windown inference display. For headless systems")
        parser.add_argument("--ext_output", action='store_true',
                            help="display bbox coordinates of detected objects")
        parser.add_argument("--config_file", default="./Yolo/cfg/yolov4.cfg",
                            help="path to config file")
        parser.add_argument("--data_file", default="./Yolo/cfg/coco.data",
                            help="path to data file")
        parser.add_argument("--thresh", type=float, default=.25,
                            help="remove detections with confidence below this value")
        return parser.parse_args()

    def check_arguments_errors(self, args):
        assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
        if not os.path.exists(args.config_file):
            raise (ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
        if not os.path.exists(args.weights):
            raise (ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
        if not os.path.exists(args.data_file):
            raise (ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
        if self.str2int(args.input) == str and not os.path.exists(args.input):
            raise (ValueError("Invalid video path {}".format(os.path.abspath(args.input))))

    def str2int(self, video_path):
        """
        argparse returns and string althout webcam uses int (0, 1 ...)
        Cast to int if needed
        """
        try:
            return int(video_path)
        except ValueError:
            return video_path

    def get_predictionDrawed(self, frame, detection):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.width, self.height),
                                   interpolation=cv2.INTER_LINEAR)
        random.seed(3)  # deterministic bbox colors
        if frame_resized is not None:
            image = darknet.draw_boxes(detection, frame_resized, self.class_colors)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        return None

