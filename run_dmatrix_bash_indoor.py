import os
import cv2
import time
import threading
import serial
import config as cf
import Adafruit_PCA9685
import numpy as np
from numpy import interp
#---------My libs---------#
from bisenet_trt import TRTSegmentor
from drive_utils.mask2bbox import mask_to_bbox
from trt_utils.segcolors import lanecolor,midcolor,colors
from drive_utils.driving import steering, findIntersect
# from drive_utils.IPM import compute_perspective_transform, compute_point_perspective_transform
from drive_utils.calib_utils_bash import distort_calib2 
from drive_utils.distance_utils import distance_finder
#---------Servo libs---------#
import Adafruit_PCA9685
#---------Run configs---------#
cf.infer = 1 # video - 0; webcam - 1
cf.show = False
cf.show_arrow = True
cf.run = True
cf.servo = 0
cf.aglservo = 0
cf.stop = False
cf.intersect = False
cf.mode1 = 0
cf.mode2 = 0
#---------BISENET LANE DETECTION---------#
def bisenet_pipeline():
    i = 0
    while True:
        ret, frame = cap.read()
        t = time.time()
        image = frame.copy()
        #---------CALIBRATE CAMERA---------#
        if cf.infer == 1:
            image = distort_calib2(image)
        # cv2.imwrite('Desktop/DEEP_CAR/data/frame'+str(i)+'.jpg', image)
        # i += 1
        image = cv2.resize(image, (360,360))
        h,w,_ = image.shape
        duration = bisenet_trt.infer(image, benchmark=True)
        lanemask, midmask, obstacle, fullmask = bisenet_trt.draw(image)
        #---------CHECK LANE INTERSECTION---------#
        cf.intersect = findIntersect(lanemask)
        #---------CALC STEERING ANGLE---------#
        cf.aglservo = steering(image, lanemask, obstacle, fullmask, None) #mask,kp,ki,kd
        # cf.aglservo  = interp(angle,[-30,30],[30,-30])
        # cf.aglservo = interp(angle,[-30,30],[500,300]) # map angle to servo
        #---------SHOW RESULTS---------#
        cv2.imshow("frame", image)
        # cv2.imshow('obstacle', obsmask)
        # if cf.show:
        #     stacked_image = dmatrix(image,fullmask)
        #     cv2.imshow('results', stacked_image)
        key = cv2.waitKey(1) 
        if key == ord('n'):
            cf.vision = True
            cf.map = False
        if key == ord('m'):
            cf.map = True
            cf.vision = False
        if key == ord('q'):
            cf.stop = True
            break
        print('FPS=:{:.2f}'.format(1/(time.time()-t)))

# CONTROL MOTOR AND SERVO
def control_pipeline():
    servolist = []
    while True:
        # READ CONTROL SIGNAL FROM ARDUINO
        s = str(arduino.readline())[2:31]
        # if s[0:4] == 'stop':
        #     print('stopped servos & motors')
        # mode1 = float(s[0:4])
        # mode2 = float(s[5:9])
        # throth = float(s[15:19])
        # servo = float(s[20:24])
        try:
            cf.mode1 = float(s[0:4])
            cf.mode2 = float(s[5:9])
            throth = float(s[15:19])
            throth = interp(throth,[1100,1900],[400,300])
            servo_f = float(s[20:24])
            servo = interp(servo_f,[1100,1900],[310,490])
            # pwm.set_pwm(0, 0, int(servo)) #295
            pwm.set_pwm(1, 0, int(throth)) #295
        except ValueError:
            pass
        with open('Desktop/DEEP_CAR/sensor_log/eyaw.txt') as file:
            lines = file.readlines()
            aglgps = (float(lines[0]))
        # MANUAL MODE
        if cf.mode2 < 1110 and cf.mode2 > 1090:
            if cf.mode1 < 1110 and cf.mode1 > 1090:
                pwm.set_pwm(0, 0, int(servo))
                print('manual mode')
                if cf.mode2 < 1910 and cf.mode2 > 1890:
                    break
        # AUTO MODE
        if cf.mode2 < 1910 and cf.mode2 > 1890:
            # STRAIGHT
            if cf.mode1 < 1110 and cf.mode1 > 1090:
                pwm.set_pwm(0,0,400)
                print('straight')
                if cf.mode1 > 1110:
                    break
            #CAMERA MODE
            if cf.mode1 < 1510 and cf.mode1 > 1490:
                if servo_f < 1520 and servo_f > 1500:
                    pwm.set_pwm(0, 0, int(interp(cf.aglservo,[-30,30],[490,310])))
                    print('servo in cam mode',servo)
                else:
                    pwm.set_pwm(0, 0, int(servo))
                print('camera mode')
                if cf.mode1 > 1510:
                    break
            #GPS MODE
            if cf.mode1 < 1910 and cf.mode1 > 1890:
                if servo_f < 1520 and servo_f > 1500:
                    pwm.set_pwm(0, 0, int(interp((-aglgps*0.5),[-30,30],[490,310])))
                else:
                    pwm.set_pwm(0, 0, int(servo))
                print('gps mode')
                if cf.mode1 < 1890:
                    break
            if cf.mode2 < 1890:
                break

        # READ GPS SIGNAL FROM ANOTHER FILE
        # CAMERA MODE
            # if cf.servo < 1510 and cf.servo > 1490:
            #     if cf.vision:
            #         # servolist.clear()
            #         combined_angle = (float(cf.aglservo)*0.1 - float(aglgps)*0.9)
            #         pwm.set_pwm(0, 0, int(interp(cf.aglservo,[-30,30],[490,310])))
            #         # os.system('cls' if os.name=='nt' else 'clear')
            #         print('camera steering:', cf.aglservo) 
            #         #print('gps steering:', aglgps)
            #         #print('combined steering:', combined_angle)
            #         # servolist.append()
            #     if cf.map:
            #         pwm.set_pwm(0, 0, int(interp((-aglgps*0.5),[-30,30],[490,310])))
            #         print('gps steering:', aglgps)

        if cf.stop == True:
            break

if __name__ == '__main__':
    # INIT VIDEO CAPTURE OPJECT
    if cf.infer == 0:
        print('infer with video')
        cap = cv2.VideoCapture('Desktop/DEEP_CAR/video/project2.avi')
    else:
        cap = cv2.VideoCapture("v4l2src device=/dev/video1 ! image/jpeg, width=640, height=360 ! jpegdec ! video/x-raw, format=I420 ! videoconvert ! appsink drop=true", cv2.CAP_GSTREAMER)
    # INIT PWM MODULE
    if cf.run:
        arduino = serial.Serial('/dev/ttyACM0', timeout=1, baudrate=9600)
        pwm = Adafruit_PCA9685.PCA9685(address=0x40, busnum=1)
        pwm.set_pwm_freq(60)
        pwm.set_pwm(1, 0, 350)
    # INIT MODEL 
    bisenet_trt=TRTSegmentor('Desktop/DEEP_CAR/checkpoints_trt/bisenet5_15_3_2.onnx', 
        colors,
        lanecolor,
        midcolor, 
        device='GPU', 
        precision='FP16',
        calibrator=None, 
        dla_core=0)
 
    # THREADED RUNS
    # lock = multiprocessing.Lock() 
    control_thread = threading.Thread(name='control', target=control_pipeline)
    control_thread.start()
    segment_thread = threading.Thread(name='bisenet', target=bisenet_pipeline())
    segment_thread.start()

    # NORMAL RUNS
    # bisenet_pipeline()
