import cv2
import numpy as np
import logging
import math
import datetime
import sys

from numpy import dtype
from torch import int32
from drive_utils.detect_mid_lane import *

_SHOW_IMAGE = True


# class HandCodedLaneFollower(object):

# 	def __init__(self, car=None):
# 		logging.info('Creating a HandCodedLaneFollower...')
# 		self.car = car
# 		self.curr_steering_angle = 90

# 	def follow_lane(self, mask1, mask2, mask3):
# 		# Main entry point of the lane follower
# 		# show_image("road_mask", mask1)
# 		# show_image("mid_mask", mask2)
# 		lane_lines, frame = detect_lane(mask1,mask2,mask3)
# 		final_frame = self.steer(frame, lane_lines)

# 		return final_frame

# 	def steer(self, frame, lane_lines):
# 		logging.debug('steering...')
# 		if len(lane_lines) == 0:
# 			logging.error('No lane lines detected, nothing to do.')
# 			return frame

# 		new_steering_angle = compute_steering_angle(frame, lane_lines)
# 		self.curr_steering_angle = stabilize_steering_angle(self.curr_steering_angle, new_steering_angle, len(lane_lines))

# 		if self.car is not None:
# 			self.car.front_wheels.turn(self.curr_steering_angle)
# 		curr_heading_image = display_heading_line(frame, self.curr_steering_angle)
# 		show_image("heading", curr_heading_image)

# 		return curr_heading_image

############################
# Frame processing steps
############################
def fill_hole(frame):
	im_floodfill=frame.copy()
	h,w=frame.shape[:2]
	mask=np.zeros((h+2,w+2),np.uint8)
	# floodfill from point (0,0)
	cv2.floodFill(im_floodfill, mask, (0,0), 255)
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)
	# Combine the two images to get the foreground 
	im_out= frame | im_floodfill_inv
	im_out=cv2.resize(im_out,(h,w))
	# Creating kernel
	kernel = np.ones((3, 3), np.uint8)
	# Using cv2.erode() method 
	image = cv2.erode(im_out, kernel)
	edges=im_out-image
	cv2.imshow('fill',edges)
	return edges

def line_detect(imge, threshold,lower_angle_lim, upper_angle_lim ):
	'''
	line_detect(imge, threshold,lower_angle_lim, upper_angle_lim ) function is used to detect lines from the processed image which have a houghscore greater than the threshold and whose angles does not lie between the lower and upper angle limit.(this is to ensure that the lines are almost vertical)

	input: 	imge - numpy.ndarray, the input image
		threshold - integer, the min threshold values which th hough score of detected lines should satisfy
		lower_angle_lim and upper_angle_lim - floats, the detected lines would not be between these angle limits.
	
	output:
		req_lines - list, a list of all lines which satisfy the required conditions

	'''
	all_lines = cv2.HoughLines(imge,1,np.pi/180,threshold)
	req_lines = []
	if isinstance(all_lines,np.ndarray):
		#ensuring that all the detected lines are nearly vertical. i.e. not lying between the two angle limits
		for each_line in all_lines:
			for rho,theta in each_line:
				if theta < np.pi*lower_angle_lim/180 or theta> np.pi*upper_angle_lim/180:
					a = np.cos(theta)
					b = np.sin(theta)
					x0 = a*rho
					y0 = b*rho
					x1 = int(x0 + 1000*(-b))
					y1 = int(y0 + 1000*(a))
					x2 = int(x0 - 1000*(-b))
					y2 = int(y0 - 1000*(a))
					req_lines.append([x1,y1,x2,y2,rho,theta])
	return req_lines

#function to detect lines by iteratively decreasing Hough threshold untill atleast one line is detected
def detect_best_lines(img,Hough_threshold_upper_limit,Hough_threshold_lower_limit,lower_angle_lim, upper_angle_lim):
	'''
	detect_best_lines(img,Hough_threshold_upper_limit,Hough_threshold_lower_limit,lower_angle_lim, upper_angle_lim) function is used to detect lines by iteratively decreasing Hough threshold untill atleast one line is detected

	input: 	img - numpy.ndarray, the image on which lines are to be detected
		Hough_threshold_upper_limit - int, The threshold with which the iterative reduction begins
		Hough_threshold_lower_limit - int, The lowest threshold to which iterative reduction is done untill a line is found
		lower_angle_lim, upper_angle_lim - floats, Angles between which the detected lines should lie for them to be considered as potential lane markings

	output: lines - list, the list of all lines that are detected at highest possible threshold
	'''
	Hough_threshold = Hough_threshold_upper_limit
	lines = line_detect(img, Hough_threshold,lower_angle_lim, upper_angle_lim )
	while (isinstance(lines,list) and Hough_threshold>Hough_threshold_lower_limit):
		lines = line_detect(img, Hough_threshold,lower_angle_lim, upper_angle_lim )
		Hough_threshold -= 5
		pass
	return lines

#function to determine intercept of a line with the bottom edge of the image
def find_intercept(height,top_margin,left_margin,x1,y1,x2,y2):
	'''
	find_intercept(height,top_margin,left_margin,x1,y1,x2,y2) function is used to determine intercepts of a given line with the bottom of the image

	input:  height - integer, height of the image on which the line lies
		top_margin,left margin - integers, the top and left margins of the region of interest in pixels
		x1,y1,x2,y2 - coordinates of two points which lie on the line for which intercepts need to be found
		
	output: intercept - float, the y coordinate where the line intersects the bottom line of the image.(origin beginning in the top left corner of the image)  
	'''
	Y1 = -(y1 + top_margin)
	X1 = x1 + left_margin
	Y2 = -(y2 + top_margin)
	X2 = x2+ left_margin
	Y = - height
	slope = float(Y2-Y1)/float(X2-X1)
	intercept = 0
	if slope != 0:
		intercept = (float(Y-Y1)/slope) + float(X1)
	return intercept


#function to determine the left and right lanes from a set of lines
def decide_lanes(left_or_right,lines,height,top_margin,left_margin,right_margin):
	'''
	decide_lanes(left_or_right,lines,height,top_margin,left_margin,right_margin) function is used to detect the left and right lanes out of all the lines detected. it uses a weighted score of distance between y intercept and mid point along with the steepness of the line to determine the lanes.

	input:  left_or_right - string, the choice of line which is to be detected i.e. "left"/"right"
		lines - list, a list of all input lines
		height - integer, heightof the image in pixels
		top_margin, left_margin, right_margin - integers, the top, left and right margins of the region of interest in pixels

	output:
		lane - list, has the structure [x1,y1,x2,y2,theta,intercept_with the bottom of the image]
	'''
	
	dist_weight = 2  	
	angle_weight = 10
	lane_score = -100000000
	lane =[]
	
	if left_or_right in ['left','Left','LEFT']:
		for x1,y1,x2,y2,rho,theta in lines:
			intercept = find_intercept(height,top_margin,left_margin,x1,y1,x2,y2) 
			if theta>0 and theta<np.pi/2: #the angle of left lane is assumed to be between 0and 90 degrees
				score = (-1*angle_weight*theta)+ (-1*dist_weight * (right_margin - intercept)) #lower the distance from center of the two lanes, higher the score, steeper the line higher the score
				if score > lane_score: #the line with highest score is the left lane
					lane_score = score
					lane = [x1+left_margin,y1+top_margin,x2+left_margin,y2+top_margin,theta,intercept]
		

	if left_or_right in ['right','Right','RIGHT']:
		for x1,y1,x2,y2,rho,theta in lines:
			intercept = find_intercept(height,top_margin,left_margin,x1,y1,x2,y2) 
			if theta < np.pi and theta > np.pi/2 : #the angle of right lane is assumed to be between 180 and 90 degrees
				score = (angle_weight*theta)+ (-1*dist_weight * (intercept - left_margin)) #lower the distance from center of two lanes, higher the score, steeper the line higher the score
				if score > lane_score: #the line with highest score is the right lane
					lane_score = score
					lane = [x1+left_margin,y1+top_margin,x2+left_margin,y2+top_margin,theta,intercept]

	return lane

#function to eliminate background using estimated states to reduce process time
def elimnate_predicted_background(lane, img, img_width,img_height,top_margin, left_margin,  height, Q_est, prediction_error_tolerance ):
	'''
	elimnate_predicted_background(lane,img,img_width,img_height,top_margin,left_margin,height,Q_est,prediction_error_tolerance) function is used to eliminate background using state estimate to improve performance. only a thin strip around the estimate of width equal to error tolerance is left

	input:  lane - string, Choice of the lane which is to be processed i.e. "left"/"right" 
		img - numpy.ndarray, the image in which the background is to be removed
		img_width - int, Width of the image, which is to be processed
		img_height - int, Height of the image, which is to be processed
		top_margin, left_margin - ints, top and left margins used to obtain the current image from the original image based on which intercepts are calculated
		height - int, height of the original image based on which intercepts are calculated 
		Q_est - numpy.ndarray, Estimated state for the current time sample
		prediction_error_tolerance - int, the tolerance in pixels afforded to the estimated state

	output: img - numpy.ndarray, the image where background is blacked out to improve performance

	'''
	x1 = Q_est[0][0]
	y1 = height
	x2 = 0
	y2 = height+(Q_est[0][0]/np.tan(Q_est[1][0]) )
	
	if (lane in ["left", "Left","LEFT"]):
		for x in range(0,img_width):
			for y in range(0,img_height):
				if y<(y1  -  top_margin - (((x+left_margin) - (x1 - prediction_error_tolerance))*(y1-y2) /(x2-x1)   )  ) or y>(y1  -  top_margin - (((x+left_margin) - (x1 + prediction_error_tolerance))*(y1-y2) /(x2-x1)   )  )   :
					img[y][x] = 0
	if (lane in ["right", "Right","RIGHT"]):
		for x in range(0,img_width):
			for y in range(0,img_height):
				if y>(y1  -  top_margin - (((x+left_margin) - (x1 - prediction_error_tolerance))*(y1-y2) /(x2-x1)   )  ) or y<(y1  -  top_margin - (((x+left_margin) - (x1 + prediction_error_tolerance))*(y1-y2) /(x2-x1)   )  )   :
					img[y][x] = 0
	

	return img

#user defined function to calculate the dot product of two matrices
def dot_prod(a, b):#had to use a user defined function as numpy.dot was unreliable!
	'''
	dot_prod(a, b) function is used to determint the dot product between the matrices a and b. However the number of columns in a has to be equal to the number of rows in b, else an exception would be raised

	input:  a,b - numpy.ndarray, the two matrices whose dot product is to be determined
		
	output: dot - numpy.ndarray, the dot product of a and b

	'''
	if(a.shape[1]!=b.shape[0]):
		raise Exception("incompatible shapes",a.shape," ",b.shape)
		return 0
	else:
		
		dot= np.ones((a.shape[0],b.shape[1]))
		for i in range(0,a.shape[0]):
			for k in range(0,b.shape[1]):
				temp=0
				for j in range(0,a.shape[1]):
					temp += a[i][j]*b[j][k]
				dot[i][k] = temp 
		return dot


#function to determine the state and covariance estimate of the system using Kalman's algorithm
def Kalman_filter_estimate(A,B,u,Ex,Q,P):
	'''
	Kalman_filter_estimate(A,B,u,Ex,Q,P) function is used to determint the state and the state covariance matrix of the described system using Kalman's algorithm

	input:  A - numpy.ndarray, State transition matrix of the system
		B - numpy.ndarray, Control correction matrix of the system
		u - numpy.ndarray, Control input to the systemz
		Ex - numpy.ndarray, State prediction error
		Q - numpy.ndarray, state of the system at the preceding time sample
		P - numpy.ndarray, state covariance matrix at the preceding time sample

	output: Q_est - numpy.ndarray, Estimated state at current time sample
		P_est - numpy.ndarray, Estimated state covariance matrix at current time sample 
	'''
	#Prediction part of Kalman Filter

	#Step 1: Est_of_x_at_t = (A * x_at_t-1) + (B * u_at_t-1)

	Q_est = dot_prod(A,Q) +  dot_prod(B,u) #prediction of state


	#Step 2: Est_of_state_cov_at_t = (A * state_cov_at_t-1 * A_transpose) + Ex

	P_est = dot_prod(dot_prod(A,P),A.transpose())+Ex #prediction of state covariance
	
	return (Q_est,P_est)

#function to determine the updated state and state covariance matrices of the system using Kalman's algorithm
def Kalman_Filter_Update(C,Ex,Ez,Q_est, P_est, z ):
	'''
	Kalman_Filter_Update(C,Ex,Ez,Q_est, P_est, z ) function is used to calculate the updated state and state covariance matix through feedback z using Kalman's algorithm

	input:  C - numpy.ndarray, Measurement conversion matrix of the system
		Ex - numpy.ndarray, State prediction error
		Ez - numpy.ndarray, Measurement error
		Q_est - numpy.ndarray, Estimated state for the current time sample
		P_est - numpy.ndarray, Estimated state covariance matrix for the current time sample
		z - numpy.ndarray, Measurement matrix

	output: Q - numpy.ndarray, Updated state of the system for the current time sample
		P - numpy.ndarray, Updated state covariance matrix for the crrent time step
	'''
	if Q_est.size:

		#Update part of Kalman filter

		#Step 3: Kalman_gain_at_t = Est_of_state_cov_at_t * C_transpose *inv( C * Est_of_state_cov_at_t * C_transpose   +  Ez  )

		K = dot_prod( dot_prod(P_est,C.transpose()), np.linalg.inv(dot_prod(C,dot_prod(P_est,C.transpose()))+Ez)  )# Kalman gain (The weightage that needs to be given for the discrepency in measurement and the prediction to update the state and its covariance )

		#Step 4: State_at_t = Est_of_state_at_t + Kalman_gain_at_t (measured_variable_at_t - (C * Est_of_state_at_t) )

		Q = Q_est + dot_prod(K , (z - dot_prod(C,Q_est)  )  ) # correcting state estimate using measured variables to obtain actual state

		#Step 5: State_cov_at_t = (I - Kalman_gain_at_t * C) Est_of_state_cov_at_t

		P = dot_prod( (np.identity(4) - dot_prod(K,C)  ), P_est)# correcting state covariance estimate using measured variables to obtain actual state covariance 

	else:
		Q = np.array([[z[0][0]],[z[1][0]],[0],[0]])#state variable [[c],[theta],[c'],[theta']]
		P = Ex #Estimate of initial state covriance matrix
	return (Q,P)



def detect_lane(mask1,mask2,fmask):
	logging.debug('detecting lane lines...')
	# State value - unchange
	top_margin = 0#pixels
	bottom_margin = 180#pixels
	left_lane_left_margin = 0#pixels
	left_lane_right_margin = 150#pixels
	right_lane_left_margin = 150#pixels
	right_lane_right_margin = 360#pixels
	intercepts = []
	predictions = []
	predictions.append(("Left_lane_intercept_estimate","Left_lane_intercept_actual", "Left_lane_intercept_measured", "Left_lane_angle_estimate","Left_lane_angle_actual", "Left_lane_angle_measured","Right_lane_intercept_estimate","Right_lane_intercept_actual", "Right_lane_intercept_measured", "Right_lane_angle_estimate","Right_lane_angle_actual", "Right_lane_angle_measured"))
	prediction_error_tolerance = 10 #pixels
	left_lane = []
	right_lane = []
	left_Q_est = np.array([])
	left_P_est = np.array([])
	right_Q_est = np.array([])
	right_P_est = np.array([])
	dt = 0.1
	u = np.array([[0.01],[0.01]])
	acc_noise = 0.1
	c_meas_noise = 0.1
	theta_meas_noise = 0.1
	Ez = np.array(   [[c_meas_noise,0],[0,theta_meas_noise]] )#measurement prediction error
	Ex = np.array( [[(dt**4)/4,0,(dt**3)/2,0],[0,(dt**4)/4,0,(dt**3)/2],[(dt**3)/2,0,(dt**2),0],[0,(dt**3)/2,0,(dt**2)]])* (acc_noise**2)#State prediction error
	#state and measurement equations
	A = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]] )
	B = np.array([[(dt**2)/2,0],[0,(dt**2)/2],[dt,0],[0,dt]] )
	C = np.array([[1,0,0,0],[0,1,0,0]] )
	# Start detect line
	laneedges = detect_edges(mask1)
	laneedges=fill_hole(laneedges)
	left_lines = detect_best_lines(laneedges,170,90,75,105)	
	left_lane = decide_lanes("left",left_lines,laneedges.shape[0],top_margin, left_lane_left_margin, left_lane_right_margin )
	print(left_lane)
	cv2.line(mask1,(left_lane[0],left_lane[1]),(left_lane[2],left_lane[3]),(255,0,0),10)
	cv2.imshow('mask',mask1)

	# lane_line_segments = detect_line_segments(laneedges)
	# mid_line_segments = detect_line_segments(midedges)

	# lane_line_segments=


	# lane_lines = average_slope_intercept(mask1, lane_line_segments)
	# # print(lane_lines)
	# mid_lines = average_slope_intercept_mid(mask2, mid_line_segments)
	# # fmask=fill_hole(fmask)
	# lane_lines_image = display_lines(fmask, lane_lines, (0, 255, 0))
	# all_lines_image = display_lines(lane_lines_image, mid_lines, (255, 0, 0))
	# # show_image("lane lines", lane_lines_image)
	# show_image("all lines", all_lines_image)

	# if mid_lines is not None:
	#     right_road,left_road=road_2_set(mask1,lane_lines,mid_lines)
	#     # print('right co',right_road)
	#     # print('left co',left_road)
	#     right_road_angle=compute_steering_angle(mask1,right_road)
	#     left_road_angle=compute_steering_angle(mask2,left_road)
	#     # print('right',right_road_angle)
	#     # print('left',left_road_angle)
	#     # print(mid_lines)
	return None

def detect_edges(frame):

	# detect edges
	edges = cv2.Canny(frame, 200, 500)

	return edges

def detect_line_segments(cropped_edges):
	# tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
	rho = 1  # precision in pixel, i.e. 1 pixel
	angle = np.pi/180  # degree in radian, i.e. 1 degree
	min_threshold = 5  # minimal of votes
	line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=3,
									maxLineGap=4)
	return line_segments


def compute_steering_angle(frame, lane_lines):
	""" Find the steering angle based on lane line coordinate
		We assume that camera is calibrated to point to dead center
	"""
	if len(lane_lines) == 0:
		logging.info('No lane lines detected, do nothing')
		return -90

	height, width, _ = frame.shape
	if len(lane_lines) == 1:
		logging.debug('Only detected one lane line, just follow it. %s' % lane_lines[0])
		x1, _, x2, _ = lane_lines[0][0]
		x_offset = x2 - x1
	else:
		_, _, left_x2, _ = lane_lines[0][0]
		_, _, right_x2, _ = lane_lines[1][0]
		camera_mid_offset_percent = 0.03 # 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
		mid = int(width / 2 * (1 + camera_mid_offset_percent))
		x_offset = (left_x2 + right_x2) / 2 - mid

	# find the steering angle, which is angle between navigation direction to end of center line
	y_offset = int(height / 2)

	angle_to_mid_radian = math.atan(x_offset / y_offset)  # angle (in radian) to center vertical line
	angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)  # angle (in degrees) to center vertical line
	steering_angle = angle_to_mid_deg + 90  # this is the steering angle needed by picar front wheel

	logging.debug('new steering angle: %s' % steering_angle)
	return steering_angle

def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=5, max_angle_deviation_one_lane=1):
	if num_of_lane_lines == 2 :
		# if both lane lines detected, then we can deviate more
		max_angle_deviation = max_angle_deviation_two_lines
	else :
		# if only one lane detected, don't deviate too much
		max_angle_deviation = max_angle_deviation_one_lane
	
	angle_deviation = new_steering_angle - curr_steering_angle
	if abs(angle_deviation) > max_angle_deviation:
		stabilized_steering_angle = int(curr_steering_angle
										+ max_angle_deviation * angle_deviation / abs(angle_deviation))
	else:
		stabilized_steering_angle = new_steering_angle
	logging.info('Proposed angle: %s, stabilized angle: %s' % (new_steering_angle, stabilized_steering_angle))
	return stabilized_steering_angle

############################
# Utility Functions
############################
def display_lines(frame, lines, line_color=(0, 255, 0), line_width=10):
	line_image = np.zeros_like(frame)
	if lines is not None:
		for line in lines:
			for x1, y1, x2, y2 in line:
				cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
	line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
	return line_image

def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5, ):
	heading_image = np.zeros_like(frame)
	height, width, _ = frame.shape

	# figure out the heading line from steering angle
	# heading line (x1,y1) is always center bottom of the screen
	# (x2, y2) requires a bit of trigonometry

	# Note: the steering angle of:
	# 0-89 degree: turn left
	# 90 degree: going straight
	# 91-180 degree: turn right 
	steering_angle_radian = steering_angle / 180.0 * math.pi
	x1 = int(width-width/4)
	y1 = height
	x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
	y2 = int(height / 2)
	# print("Steering Angle: "+ str(steering_angle))
	cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
	heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
	return heading_image

def length_of_line_segment(line):
	x1, y1, x2, y2 = line
	return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def show_image(title, frame, show=_SHOW_IMAGE):
	if show:
		cv2.imshow(title, frame)


def make_points(frame, line):
	height, width, _ = frame.shape
	slope, intercept = line
	y1 = height  # bottom of the frame
	y2 = int(y1 * 1/9)  # make points from middle of the frame down
	if intercept==0:
		intercept=200
	if slope==0:
		slope=-0.1
	# bound the coordinates within the frame
	x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
	x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
	return [[x1, y1, x2, y2]]

def float_to_int(x):
	if x == float('inf') or x == float('-inf'):
		return float(640) # or a large value you choose
	return int(x)

############################
# Test Functions
############################
# def test_photo(test_image):
# 	land_follower = HandCodedLaneFollower()
# 	#frame = cv2.imread(test_image)
# 	combo_image = land_follower.follow_lane(test_image)
# 	show_image('final', combo_image, True)

# middle_lane=[]
####################
def road_2_set(frame,lane_lines,mid_lane):
	global left_lane, right_lane
	# find middle points of two lanes
	right_road=[]
	left_road=[]
	print(len(lane_lines))
	if len(lane_lines)==2:
		left_lane=lane_lines[0]
		right_lane=lane_lines[1]

		if lane_lines[0] is not None:
			left_road.append(left_lane)
		if len(mid_lane)==1:
			left_road.append(mid_lane[0])
		if len(mid_lane)==1:
			right_road.append(mid_lane[0])
		if lane_lines[1] is not None:
			right_road.append(right_lane)
	# print('right road',right_road)
	# print('left road',left_road)
	return right_road,left_road



	