from numpy import full
import sys
import numpy as np
import cv2
import time
import threading
import config as cf
from numpy import interp
#---------My libs---------#
from bisenet_trt import TRTSegmentor
from drive_utils.mask2bbox import mask_to_bbox
from trt_utils.segcolors import lanecolor,midcolor,colors
from drive_utils.driving import steering
# from drive_utils.drivingv2 import *
# from drive_utils.IPM import compute_perspective_transform, compute_point_perspective_transform
# import drive_utils.drivingv3 as st_utils
# import drive_utils.drivingv4 as st_utils
# from drive_utils.drivingv5 import *
from drive_utils.calib_utils import distort_calib2 
from drive_utils.distance_utils import distance_finder
#---------Servo libs---------#
import Adafruit_PCA9685
#---------Run configs---------#
cf.infer = 1 # video - 0; webcam - 1
cf.show = False
cf.show_arrow = True
cf.run = True
#---------BISENET LANE DETECTION---------#
def bisenet_pipeline():
    # land_follower = st_utils.HandCodedLaneFollower()
    if cf.run:
        pwm = Adafruit_PCA9685.PCA9685(address=0x40, busnum=1)
        pwm.set_pwm_freq(60)
    while(cap.isOpened()):
        ret, frame = cap.read()
        t = time.time()
        image = frame.copy()
        #---------CALIBRATE CAMERA---------#
        if cf.infer == 1:
            image = distort_calib2(image)
        else: pass
        image = cv2.resize(image, (360,360))
        h,w,_ = image.shape
        duration = bisenet_trt.infer(image, benchmark=True)
        lanemask, midmask, obstacle, fullmask = bisenet_trt.draw(image)
        #---------MASK TO BBOX---------#
        dis = []
        # obsmask = cv2.cvtColor(obstacle, cv2.COLOR_BGR2GRAY)
        # if 255 in obsmask:
        #     obsbbox = mask_to_bbox(obsmask)
        #     for box in obsbbox:
        #         FL = 250
        #         WIDTH = box[2]-box[0]
        #         KNOWNWIDTH = 1.86
        #         distance = np.round(distance_finder(FL, WIDTH, KNOWNWIDTH), 2)
        #         dis.append(distance)
        #         cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255,0,0), 2)
        #         cv2.putText(image, str(distance)+'m', (int(box[0]),int(box[3])+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 1, cv2.LINE_AA)
        # else: dis=[0]
        # closest_obs = min(dis)
        # print(closest_obs)
        #---------CALC STEERING ANGLE---------#
        angle = steering(image, lanemask, obstacle, fullmask, None) #mask,kp,ki,kd
        sangle  = interp(angle,[-30,30],[30,-30])
        aglservo = interp(angle,[-30,30],[500,300]) # map angle to servo
        #---------READ GPS SIGNALS FROM TXT FILE---------#
        with open('sensor_log/eyaw.txt') as file:
            lines = file.readlines()
        print('eyaw from gps',lines)
        print('eyaw from vision',sangle)
        #---------SEND ANGLE TO SERVO---------#
        if cf.run:
            pwm.set_pwm(0, 0, int(aglservo))
            print('steering:', aglservo)
        #---------SHOW RESULTS---------#
        cv2.imshow("frame", image)
        # cv2.imshow('obstacle', obsmask)
        # if cf.show:
        #     stacked_image = dmatrix(image,fullmask)
        #     cv2.imshow('results', stacked_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print('FPS=:{:.2f}'.format(1/(time.time()-t)))
        

if __name__ == '__main__':
    # INIT VIDEO CAPTURE OPJECT
    if cf.infer == 0:
        cap = cv2.VideoCapture('video/project2.avi')
    else:
        cap = cv2.VideoCapture("v4l2src device=/dev/video1 ! image/jpeg, width=640, height=360 ! jpegdec ! video/x-raw, format=I420 ! videoconvert ! appsink drop=true", cv2.CAP_GSTREAMER)
    bisenet_trt=TRTSegmentor('checkpoints_trt/bisenet5_15_3_2.onnx', 
        colors,
        lanecolor,
        midcolor, 
        device='GPU', 
        precision='FP16',
        calibrator=None, 
        dla_core=0)

    # THREADED RUNS
    # hough_thread = threading.Thread(name='hough_trans', target=hough_pipeline)
    # hough_thread.start()
    # segment_thread = threading.Thread(name='lane_detect', target=bisenet_pipeline())
    # segment_thread.start()

    # NORMAL RUNS
    bisenet_pipeline()
