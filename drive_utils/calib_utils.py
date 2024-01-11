from xml.sax import handler
import cv2
import numpy as np
from numpy import loadtxt

mtx=[]
with open('Desktop/DEEP_CAR/drive_utils/camera_matrix.txt', 'r') as f:
    for line in f:
        mtx.append(list(map(float,line.split())))
mtx=np.asarray(mtx)
dist=[]
with open('Desktop/DEEP_CAR/drive_utils/distortion_coefficients.txt', 'r') as f:
    for line in f:
        dist.append(list(map(float,line.split())))
dist=np.asarray(dist)

def distort_calib(img, balance=0.1, dim2=None, dim3=None):
    print(img.shape)
    # img = cv2.imread(img_path)
    # dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    # assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    # if not dim2:              
    #     dim2 = dim1
    # if not dim3:
    #     dim3 = dim1
    # scaled_K = mtx* dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    # scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    # new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, dist, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.initUndistortRectifyMap(mtx, dist, np.eye(3), mtx, DIM, cv2.CV_32FC1)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def distort_calib2(img):
    h,w,_=img.shape
    # print(img.shape)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),0,(w,h))
    # # Checking to make sure the new camera materix was properly generated
    # print(newcameramtx)
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, np.eye(3), newcameramtx, (w,h), cv2.CV_16SC2)

    # # Undistorting
    # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    image = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    x,y,w,h=roi
    image=image[y:y+h,x:x+w]
    # print(image.shape)
    return image
