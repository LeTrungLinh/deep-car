from numpy import full
import torch
import numpy as np
import cv2
import time
import threading
import config as cf
from PIL import Image
from torchvision import transforms
#---------My libs---------#
from bisenet_trt import TRTSegmentor
from trt_utils.segcolors import lanecolor,midcolor,colors
from drive_utils.drivingv2 import *
from drive_utils.IPM import compute_perspective_transform, compute_point_perspective_transform
# import drive_utils.drivingv3 as st_utils
import drive_utils.drivingv4 as st_utils
from drive_utils.drivingv5 import *
from drive_utils.calib_utils import distort_calib2 

# cuDnn configurations
# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
# from adafruit_servokit import ServoKit
# kit = ServoKit(channels=16)
cf.warped_lane = None

#---------BISENET LANE DETECTION---------#
def bisenet_pipeline():
    land_follower = st_utils.HandCodedLaneFollower()
    # while(cap.isOpened()):
    # ret, frame = cap.read()
    mask = cv2.imread('1000.png')
    t = time.time()
    # image = distort_calib2(image)
    mask = cv2.resize(mask, (360,360))
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    # duration = bisenet_trt.infer(image, benchmark=True)
    # lanemask, midmask, obsmask, fullmask = bisenet_trt.draw(image)
    # warped_lane = Perspective_transformed(lanemask, 360, 360)
    # warped_mid = Perspective_transformed(midmask, 360, 360)
    # warped_full = Perspective_transformed(fullmask, 360, 360)
    cv2.imshow('road', mask)
    # cv2.imshow('mid', midmask)
    # cv2.imshow('obs', obsmask)
    # stack, mask = dmatrix(undistorted,lanemask)
    # cv2.imshow('results', stack)
    # HOUGH TRANSFORM LANE DETECTION
    combo_image = land_follower.follow_lane(mask,mask,mask)
    cv2.imshow('combo', combo_image)
    # lane_image= detect_lane(warped_lane,warped_mid,warped_full)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        
    print('FPS=:{:.2f}'.format(1/(time.time()-t)))

        # SEND ANGLE TO SERVO

        # kit.servo[0].angle=angle
        # print('steering:', steering)

	
def hough_pipeline():
    while True:
        print(cf.warped_lane)


if __name__ == '__main__':
    # INIT VIDEO CAPTURE OPJECT
    # cap = cv2.VideoCapture('video/project2.avi')
    # cap = cv2.VideoCapture("v4l2src device=/dev/video1 ! image/jpeg, width=640, height=360 ! jpegdec ! video/x-raw, format=I420 ! videoconvert ! appsink drop=true", cv2.CAP_GSTREAMER)
    # cap = cv2.VideoCapture(2)    
    # bisenet_trt=TRTSegmentor('checkpoints_trt/bisenet5_15_3_2.onnx', 
    #     colors,
    #     lanecolor,
    #     midcolor, 
    #     device='GPU', 
    #     precision='FP16',
    #     calibrator=None, 
    #     dla_core=0)


    # NORMAL RUNS
    bisenet_pipeline()
