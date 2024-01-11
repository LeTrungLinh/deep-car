import cv2
import time
from calib_utils import distort_calib2 

def bisenet_pipeline():
    cap = cv2.VideoCapture("v4l2src device=/dev/video0 ! image/jpeg, width=640, height=360 ! jpegdec ! video/x-raw, format=I420 ! videoconvert ! appsink drop=true", cv2.CAP_GSTREAMER)
    i = 0
    while True:
        t = time.time()
        ret, frame = cap.read()
        image = frame.copy()
        #---------CALIBRATE CAMERA---------#
        image = distort_calib2(image)
        cv2.imwrite('Desktop/DEEP_CAR/data/afternoon'+str(i)+'.jpg', image)
        i += 1

if __name__ == '__main__':
    bisenet_pipeline()