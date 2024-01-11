import torch
import numpy as np
import cv2
import time
import threading
from PIL import Image
from torchvision import transforms
#---------My libs---------#
from bisenet_trt import TRTSegmentor
from trt_utils.segcolors import colors
from drive_utils.driving import dmatrix, steering
from drive_utils.drivingv2 import *
import drive_utils.drivingv3 as st_utils
from gps_imu import client, simulation
# cuDnn configurations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
# from adafruit_servokit import ServoKit
# kit = ServoKit(channels=16)


#---------BISENET LANE DETECTION---------#
def bisenet_pipeline():
    ret, frame = cap.read()
    land_follower = st_utils.HandCodedLaneFollower()
    fps=0.0
    while ret:
        image = frame.copy()
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        duration=bisenet_trt.infer(image, benchmark=True)
        drawn,mask=bisenet_trt.draw(image)
        # cv2.imshow('segmented', drawn)
        # cv2.imshow('mask-colored', mask)
        k=cv2.waitKey(1)
        if k==ord('q'):
            break
        fps=0.9*fps+0.1/(duration)
        print('FPS=:{:.2f}'.format(fps))
        ret,frame=cap.read()

        # HOUGH TRANSFORM LANE DETECTION
        combo_image = land_follower.follow_lane(mask,image)

        # SEND ANGLE TO SERVO
        # kit.servo[0].angle=angle
        # print('steering:', steering)
        # cv2.imshow("frame", mask)

	
def yolo_pipeline():
    pass


if __name__ == '__main__':
    # INIT VIDEO CAPTURE OPJECT
    # cap = cv2.VideoCapture(" v4l2src device=/dev/video1 ! image/jpeg, format=MJPG, width=1280, height=720 ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1", cv2.CAP_GSTREAMER)
    cap = cv2.VideoCapture('video/project2.avi')
    # INIT TRT MODELS
    bisenet_trt=TRTSegmentor('checkpoints_trt/bisenet110.onnx', 
        colors, 
        device='GPU', 
        precision='FP16',
        calibrator=None, 
        dla_core=0)

    # THREADED RUNS
    lock = threading.Lock()  
    lanedet_thread = threading.Thread(name='lane_detect', target=bisenet_pipeline())
    client_thread = threading.Thread(target = client, args = (lock,'sensor_log/log_file.csv'))
    animation_thread = threading.Thread(target = simulation, args = (lock,))
    lanedet_thread.start()
    client_thread.start()
    animation_thread.start()

    client_thread.join()
    lanedet_thread.join()