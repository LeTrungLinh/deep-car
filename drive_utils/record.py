import time
import numpy as np
import cv2
from calib_utils import distort_calib2

cap = cv2.VideoCapture("v4l2src device=/dev/video1 ! image/jpeg, width=640, height=360 ! jpegdec ! video/x-raw, format=I420 ! videoconvert ! appsink drop=true", cv2.CAP_GSTREAMER)
# cap=cv2.VideoCapture('video/vid3.mp4')
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
size = (640,360)
out = cv2.VideoWriter('Desktop/DEEP_CAR/video/vid_13_7.avi',fourcc, 30, size)

while(cap.isOpened()):
    t = time.time()
    ret, frame = cap.read()
    # frame = cv2.resize(frame, (640,480))
    image=frame.copy()
    image=distort_calib2(image)
    # write the flipped frame
    out.write(image)
    #cv2.waitKey(35)
    cv2.imshow('frame',image)
    print('FPS=:{:.2f}'.format(1/(time.time()-t)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()

cv2.destroyAllWindows()
