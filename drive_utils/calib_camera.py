import numpy as np
import cv2
import time 
from drive_utils.calib_utils import distort_calib2,distort_calib

cap = cv2.VideoCapture("v4l2src device=/dev/video1 ! image/jpeg, width=640, height=360 ! jpegdec ! video/x-raw, format=I420 ! videoconvert ! appsink", cv2.CAP_GSTREAMER)

while(cap.isOpened()):
    t=time.time()
    ret, frame = cap.read()
    image = frame.copy()
    print(image.shape)
    cv2.imshow('org',image)
    undistorted = distort_calib2(image)
    cv2.imshow('BB',undistorted)
    # write the flipped frame
    # out.write(frame)
    #cv2.waitKey(35)
    print("fps",1/(time.time()-t))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release everything if job is finished
cap.release()
#out.release()

cv2.destroyAllWindows()
