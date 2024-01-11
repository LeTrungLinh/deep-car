from inspect import stack
from numpy import interp
from torch._C import Stream
from PIL import Image
import math
import numpy as np
import cv2
import argparse
import time
import config as cf

image_width=320
image_height=240
last_itr=0
prev_I = 0
prev_error = 0
constrain_angle=60

deg_5 = np.zeros((240, 320))
cv2.line(deg_5, (180, 0), (160, 240), (1), 2)
cv2.line(deg_5, (140, 0), (160, 240), (1), 2)

deg_15 = np.zeros((240, 320))
cv2.line(deg_15, (224, 0), (160, 240), (1), 2)
cv2.line(deg_15, (95, 0), (160, 240), (1), 2)

deg_25 = np.zeros((240, 320))
cv2.line(deg_25, (271, 0), (160, 240), (1), 2)
cv2.line(deg_25, (48, 0), (160, 240), (1), 2)

deg_35 = np.zeros((240, 320))
cv2.line(deg_35, (328, 0), (160, 240), (1), 2)
cv2.line(deg_35, (-8, 0), (160, 240), (1), 2)

deg_45 = np.zeros((240, 320))
cv2.line(deg_45, (400, 0), (160, 240), (1), 2)
cv2.line(deg_45, (-80, 0), (160, 240), (1), 2)

deg_55 = np.zeros((240, 320))
cv2.line(deg_55, (502, 0), (160, 240), (1), 2)
cv2.line(deg_55, (-182, 0), (160, 240), (1), 2)

deg_65 = np.zeros((240, 320))
cv2.line(deg_65, (530, 0), (160, 240), (1), 2)
cv2.line(deg_65, (-210, 0), (160, 240), (1), 2)

def millis():
    return int(round(time.time() * 1000))
last_itr =millis()

def distance_matrix(road):
	global c
	intersections = np.empty((7, 4), np.uint16)
	road = (road * 255).astype(np.uint8)
	ctns,hierachy = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	if len(ctns) != 0:
		c = max(ctns, key = cv2.contourArea)
	zeros = np.zeros(road.shape)
	zeros = cv2.drawContours(zeros, c, -1, (1), 2)
	
	zeros[220:, :] = 0
	
	#cv2.imshow("ad",zeros)
	zerosa=(zeros * 255).astype(np.uint8)
	#cv2.imwrite("contour1.jpg",zerosa)
	intersections[0] = find_intersection(zeros, deg_5)
	intersections[1] = find_intersection(zeros, deg_15)
	intersections[2] = find_intersection(zeros, deg_25)
	intersections[3] = find_intersection(zeros, deg_35)
	intersections[4] = find_intersection(zeros, deg_45)
	intersections[5] = find_intersection(zeros, deg_55)
	intersections[6] = find_intersection(zeros, deg_65)
	return intersections, zerosa

def find_intersection(image, deg):
		inters = np.empty((4))
		finish = 0 #0 is found none, 1 is found right, -1 is found left
		intersect = image * deg
		coor = np.array(np.where(intersect == 1))
		# print("///////",coor.shape)
		for i in range(coor.shape[1] - 1, -1, -1):
			#print("cor")
			#print(coor[0, i], coor[1, i])
			if finish == 0:
				if coor[1, i] < image_width/2:
					inters[0] = coor[1, i]
					inters[1] = coor[0, i]
					finish = -1
				elif coor[1, i] >= image_width/2:
					inters[2] = coor[1, i]
					inters[3] = coor[0, i]
					finish = 1
			elif finish == 1:
				if coor[1, i] < image_width/2:
					inters[0] = coor[1, i]
					inters[1] = coor[0, i]
					break
			elif finish == -1:
				if coor[1, i] >= image_width/2:
					inters[2] = coor[1, i]
					inters[3] = coor[0, i]
					break
		
		return inters

def error_matrix_method(img, road_mask):
	global stack_h

	# if put_mask:
	# if dodge:
	# 	weights = [0.5, 0.4, 1.1, 0.6, 0.7, 0]
	# else: 
	weights = [0.0, 0.2, 0.5, 0.6, 0.7, 0]
	# else:
	# weights = [0.1, 0.3, 0.7, 1.1, 1.5, 0.1, 0.0]

	road_mask[road_mask < 0.52] = 0
	matrix,zerosa = distance_matrix(road_mask)

	#-----COMPUTE HEADING ERROR-----#
	matrix = matrix.astype(np.int64)
	error = 0
	for i in range(len(weights)):
		error += (matrix[i, 1] - matrix[i, 3]) * weights[i]

	#-----SHOW MATRIX POINTS ON IMAGE-----#
	img = cv2.resize(img, (320, 240))
	orgi = img.copy()
	mask = cv2.resize(mask, (320, 240))
	orgm = mask.copy()
	# mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
	cv2.circle(mask,(matrix[0][0],matrix[0][1]),3,(255,255,255),3)
	cv2.circle(mask,(matrix[0][2],matrix[0][3]),3,(255,255,255),3)
	cv2.circle(mask,(matrix[1][0],matrix[1][1]),3,(255,255,255),3)
	cv2.circle(mask,(matrix[1][2],matrix[1][3]),3,(255,255,255),3)
	cv2.circle(mask,(matrix[2][0],matrix[2][1]),3,(255,255,255),3)
	cv2.circle(mask,(matrix[2][2],matrix[2][3]),3,(255,255,255),3)
	cv2.circle(mask,(matrix[3][0],matrix[3][1]),3,(255,255,255),3)
	cv2.circle(mask,(matrix[3][2],matrix[3][3]),3,(255,255,255),3)
	cv2.circle(mask,(matrix[4][0],matrix[4][1]),3,(255,255,255),3)
	cv2.circle(mask,(matrix[4][2],matrix[4][3]),3,(255,255,255),3)
	cv2.circle(mask,(matrix[5][0],matrix[5][1]),3,(255,255,255),3)
	cv2.circle(mask,(matrix[5][2],matrix[5][3]),3,(255,255,255),3)
	# print(int(-error*2))
	# cv2.circle(img,(int(-error*2), 180),3,(255,0,0),3)
	# cv2.line(img, (int(image_width/2), image_height), (int(image_width/2), 0), (0), 2)
	# cv2.imshow('intersection',img)
	# cv2.imshow('contour',zeros)
	# zeros_t = cv2.cvtColor(zerosa,cv2.COLOR_GRAY2BGR)
	# stack_h = np.hstack((orgi, mask, zeros_t, img))

	return error * 0.5 - 10, mask

def calc_pid(error, kP, kI, kD):
	'''
	Return the calculated angle base on PID controller
	'''
	global last_itr,prev_I,prev_error
	if last_itr == 0:
		last_itr = millis()
		return 0
	else:
		itr = millis() - last_itr
		i_error = error + prev_I / itr
		d_error = (error - prev_error) / itr

		last_itr = itr
		prev_I = i_error
		prev_error = error
		pid_value = kP * error + kI * i_error + kD * d_error

		# print('Raw pid: {}'.format(pid_value))
		#pid_value = abs(pid_value)
		pid_value = np.clip(pid_value, -30, 30)
		#print('clipped pid', pid_value)
		# pid_value = interp(pid_value,[-60,60],[140,60])

		#print('goc lai', pid_value)
		return pid_value

def steering(image, mask):
	mask = cv2.cvtColor(cv2.resize(mask, (320, 240)), cv2.COLOR_BGR2GRAY)
	error, stack_img = error_matrix_method(image, mask)
	#print("erroe PID",error)
	angle = calc_pid(error, 0.5, 0.0, 600.0)
	# if cf.record:
	# 	cv2.line(stack_img, (160,240), (160,0), (0,0,0), 2)
	# cv2.imshow('ROI', cropped)
	#print("angle",angle)
	return error, angle, stack_img

def show_arrow(angle, image, color):
	steer = np.deg2rad(angle) 
	x0, y0 = 160, 240
	x1, y1 = 160, 180
	x2 = int(((x1-x0) * math.cos(steer)) - ((y1-y0) * math.sin(steer)) + x0)
	y2 = int(((x1-x0) * math.sin(steer)) + ((y1-y0) * math.cos(steer)) + y0)
	cv2.arrowedLine(image, pt1=(x0,y0), pt2=(x2,y2), color=color, thickness=5, tipLength=0.2)

def region_of_interest(mask):
    height, width = mask.shape
    frame = np.zeros_like(mask)
    # only focus bottom half of the screen 
    polygon = np.array([[
        (width / 3, height * 1 / 4),
        (width - width / 3, height * 1 / 4),
        (width, height - height / 2 - 20),
		(width, height),
		(0, height),
		(0, height - height / 2 - 20),
    ]], np.int32)

    cv2.fillPoly(frame, polygon, 255)
    masked_image = cv2.bitwise_and(mask, frame)
    return masked_image

def findIntersect(mask):
	mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
	mask = (mask * 255).astype(np.uint8)
	contours, ret = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	isIntersect=False

	best = -1
	maxsize = -1
	count = 0
	for cnt in contours:
		if cv2.contourArea(cnt) > maxsize :
			maxsize = cv2.contourArea(cnt)
			best = count
		count = count + 1

	best_cnt=contours[best]
	size_best=cv2.contourArea(best_cnt)
	# print("size_best",size_best)
	if (size_best/1000)>47.7:
		# print("intersect_here")
		isIntersect=True
	else:
		isIntersect=False
	return isIntersect
# test driving - uncomment if you run run.py
# img = cv2.imread('data/data_801.png',cv2.IMREAD_COLOR)
# # img = cv2.resize(img, (320, 240))

# mask = cv2.imread('label/data_801.png')
# # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
# # mask = cv2.resize(mask, (320, 240))
# driving(img, mask)
# steering(mask)


# #cv2.imwrite("distance_matrix1.jpg",img)
# #cv2.imshow("rgb",img)

# if cv2.waitKey(0) & 0xFF == ord('q'):

# 	cv2.destroyAllWindows()
