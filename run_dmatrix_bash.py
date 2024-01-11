import cv2
import time
import threading
import math
import serial
import config as cf
import Adafruit_PCA9685
import numpy as np
from numpy import interp
import pandas as pd
from pandas import *
#---------My libs---------#
# from drive_utils.mask2bbox import mask
from bisenet_trt import TRTSegmentor
# from drive_utils.mask2bbox import mask_to_bbox
from trt_utils.segcolors import lanecolor,midcolor,colors
from drive_utils.driving import steering, show_arrow
# from drive_utils.IPM import compute_perspective_transform, compute_point_perspective_transform
from drive_utils.calib_utils_bash import distort_calib2 
from drive_utils.distance_utils import distance_finder
#---------Servo libs---------#
import Adafruit_PCA9685
#---------Run configs---------#
cf.infer = 0 # video - 0; webcam - 1
cf.show = True
cf.show_arrow = True
cf.run = True
cf.record = False
cf.capture = True
cf.error = 0
cf.aglservo = 0
cf.aglgps = 0
cf.stop = False
cf.t = 0
cf.i = 0
cf.mode1 = 0
cf.mode2 = 0
cf.mode3 = 0
cf.lane = 'mid'
#---------BISENET LANE DETECTION---------#
def bisenet_pipeline():
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        tprev = time.time()
        image = frame.copy()
        #---------CALIBRATE CAMERA---------#
        if cf.infer == 1:
            image = distort_calib2(image)
        if cf.capture:
            if cf.mode2 < 1910 and cf.mode2 > 1890:
                if cf.mode1 < 1110 and cf.mode1 > 1090:
                    cv2.imwrite('Desktop/DEEP_CAR/data/afternoon/afternoon_15_7'+str(i+1000)+'.jpg', image)
                    i += 1
        image = cv2.resize(image, (360,360))
        h,w,_ = image.shape
        duration = bisenet_trt.infer(image, benchmark=True)
        lanemask, obstacle, fullmask = bisenet_trt.draw(image)
        #---------CHECK LANE INTERSECTION---------#
        # cf.intersect = findIntersect(lanemask)
        #---------CALC STEERING ANGLE---------#
        cf.error, cf.aglservo, stack = steering(image, lanemask, obstacle, fullmask, None) #mask,kp,ki,kd
        #---------SHOW RESULTS---------#
        if cf.show_arrow:
            show_arrow(cf.aglservo, stack, (100,0,0))
            show_arrow(-cf.aglgps, stack, (0,100,0))
            cv2.putText(stack, cf.lane, (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("stacked", cv2.resize(stack, (360,360)))
        key = cv2.waitKey(1) 
        if key == ord('q'):
            cf.stop = True
            break
        # print('FPS=:{:.2f}'.format(1/(time.time()-tprev)))
        cf.i += 1

# CONTROL MOTOR AND SERVO
def control_pipeline():
    verror = []
    vsteer = []
    gsteer = []
    t=[]
    while True:
        # READ CONTROL SIGNAL FROM ARDUINO
        s = str(arduino.readline())[2:31]
        try:
            cf.mode1 = float(s[0:4])
            cf.mode2 = float(s[5:9])
            cf.mode3 = float(s[10:14])
            throth = float(s[15:19])
            throth = interp(throth,[1100,1900],[250,450])
            # print('throth:',throth)
            servo_f = float(s[20:24])
            servo = interp(servo_f,[1100,1900],[310,490])
            # print('servo',servo)
            # pwm.set_pwm(0, 0, int(servo)) #295
            pwm.set_pwm(1, 0, int(throth)) #295
            # READ GPS FROM FILE
            with open('Desktop/DEEP_CAR/sensor_log/eyaw.txt') as file:
                lines = file.readlines()
                cf.aglgps = (float(lines[0]))
            #RECORD MODE
            if cf.record:
                # print('recording', cf.i)
                if cf.i % 10 == 0:
                    cf.t += 1
                    # print('saved')
                    verror.append(cf.error)
                    vsteer.append(cf.aglservo)
                    gsteer.append(-cf.aglgps)
                    t.append(cf.t)
                    print('value:', str(cf.t), str(cf.aglservo), str(-cf.aglgps))
                    df = pd.DataFrame({'time':t, 'verror':verror, 'vsteer':vsteer, 'gsteer':gsteer})
                    df.to_csv('Desktop/DEEP_CAR/sensor_log/steer.csv', index=False)
        except Exception as e:
            print("type error: " + str(e))
            pass
        # MANUAL MODE
        if cf.mode2 < 1110 and cf.mode2 > 1090:
            # if cf.mode1 < 1110 and cf.mode1 > 1090:
            pwm.set_pwm(0, 0, int(servo))
            print('manual mode')
        # LANE CHANGE
        if cf.mode3 > 1600:
            cf.lane = 'left'
        elif cf.mode3 < 1200:
            cf.lane = 'right'
        # AUTOMODE
        if cf.mode2 < 1910 and cf.mode2 > 1890:
            # STRAIGHT
            if cf.mode1 < 1110 and cf.mode1 > 1090:
                cf.lane = 'mid'
                pwm.set_pwm(0, 0, int(servo))
                print('recording...')
                # if cf.mode1 > 1110:
                #     break
            #CAMERA MODE
            if cf.mode1 < 1510 and cf.mode1 > 1490:
                if servo_f < 1530 and servo_f > 1510:
                    pwm.set_pwm(0, 0, int(interp(cf.aglservo,[-30,30],[490,310])))
                    # print('servo in cam mode',servo)
                else:
                    pwm.set_pwm(0, 0, int(servo))
                print('camera mode')
                # if cf.mode1 > 1510:
                #     break
            #GPS MODE
            if cf.mode1 < 1910 and cf.mode1 > 1890:
                print(servo_f)
                if servo_f < 1530 and servo_f > 1510:
                    pwm.set_pwm(0, 0, int(interp((-cf.aglgps*0.8),[-30,30],[490,310])))
                else:
                    pwm.set_pwm(0, 0, int(servo))
                print('gps mode')
                # if cf.mode1 < 1890:
                #     break
            # if cf.mode2 < 1890:
            #     break
            
        if cf.stop == True:
            break

if __name__ == '__main__':
    # INIT MODEL 
    bisenet_trt=TRTSegmentor('Desktop/DEEP_CAR/checkpoints_trt/bisenet18_2_7.onnx', 
        colors,
        lanecolor,
        midcolor, 
        device='GPU', 
        precision='FP16',
        calibrator=None, 
        dla_core=0)
    # INIT VIDEO CAPTURE OPJECT
    video = 'Desktop/DEEP_CAR/video/project2.avi'
    source0 = "v4l2src device=/dev/video0 ! image/jpeg, width=640, height=360 ! jpegdec ! video/x-raw, format=I420 ! videoconvert ! appsink drop=true"
    source1 = "v4l2src device=/dev/video1 ! image/jpeg, width=640, height=360 ! jpegdec ! video/x-raw, format=I420 ! videoconvert ! appsink drop=true"
    if cf.infer == 0:
        print('infer with video')
        cap = cv2.VideoCapture(video)
    else:
        print('infer with webcam')
        cap = cv2.VideoCapture(source0, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap = cv2.VideoCapture(source1, cv2.CAP_GSTREAMER)
        # cap = cv2.VideoCapture(0, CAP_V4L)
    # INIT PWM MODULE
    if cf.run:
        arduino = serial.Serial('/dev/ttyUSB0', timeout=1, baudrate=9600)
        pwm = Adafruit_PCA9685.PCA9685(address=0x40, busnum=1)
        pwm.set_pwm_freq(60)
        pwm.set_pwm(1, 0, 350)
 
    # THREADED RUNS
    # lock = multiprocessing.Lock()
    control_thread = threading.Thread(name='control', target=control_pipeline)
    control_thread.start()
    segment_thread = threading.Thread(name='bisenet', target=bisenet_pipeline())
    segment_thread.start()
    #control_thread.join()
    
    # NORMAL RUNS
    # bisenet_pipeline()
