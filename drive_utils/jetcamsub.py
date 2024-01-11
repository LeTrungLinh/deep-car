from jetcam.usb_camera import USBCamera
from calib_utils import distort_calib2

import cv2
import time
camera = USBCamera(width=640, height=360, capture_width=640, capture_height=360, capture_device=1)
while True:
    t = time.time()
    image = camera.read()
    image=distort_calib2(image)
    cv2.imshow('shit', image)
    print(image.shape)
    print('FPS=:{:.2f}'.format(1/(time.time()-t)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
