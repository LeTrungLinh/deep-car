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


class HandCodedLaneFollower(object):

	def __init__(self, car=None):
		logging.info('Creating a HandCodedLaneFollower...')
		self.car = car
		self.curr_steering_angle = 90

	def follow_lane(self, mask1, mask2, mask3):
		# Main entry point of the lane follower
		# show_image("road_mask", mask1)
		# show_image("mid_mask", mask2)
		lane_lines, frame = detect_lane(mask1, mask2, mask3)
		final_frame = self.steer(frame, lane_lines)

		return final_frame

	def steer(self, frame, lane_lines):
		logging.debug('steering...')
		if len(lane_lines) == 0:
			logging.error('No lane lines detected, nothing to do.')
			return frame

		new_steering_angle = compute_steering_angle(frame, lane_lines)
		self.curr_steering_angle = stabilize_steering_angle(
			self.curr_steering_angle, new_steering_angle, len(lane_lines))

		if self.car is not None:
			self.car.front_wheels.turn(self.curr_steering_angle)
		curr_heading_image = display_heading_line(
			frame, self.curr_steering_angle)
		show_image("heading", curr_heading_image)

		return curr_heading_image

############################
# Frame processing steps
############################


def detect_lane(mask1, mask2, fmask):
	logging.debug('detecting lane lines...')
	req_lines = []
	laneedges = detect_edges(mask1)
	# laneedges = fill_hole(laneedges)
	# midedges = detect_edges(mask2)
	show_image('edges', laneedges)

	# cropped_lane = region_of_interest(laneedges)
	# cropped_mid = region_of_interest(midedges)
	# cropped_mid = cv2.dilate(cropped_mid, np.ones((5,5),np.uint8), iterations=1)
	#show_image('edges cropped', cropped_edges)

	# lane_line_segments = detect_line_segments_norm(laneedges, req_lines)
	lane_line_segments = detect_line_segments(laneedges)
	# mid_line_segments = detect_line_segments(midedges)

	# line_segment_image = display_lines(mask1, lane_line_segments)
	# mid_segment_image = display_lines(mask2, mid_line_segments)
	# show_image("line segments", line_segment_image)
	# show_image("mid segments", mid_segment_image)

	lane_lines = average_slope_intercept(mask1, lane_line_segments)
	# print(lane_lines)
	# mid_lines = average_slope_intercept_mid(mask2, mid_line_segments)
	# fmask=fill_hole(fmask)
	lane_lines_image = display_lines(fmask, lane_lines, (0, 255, 0))
	# all_lines_image = display_lines(lane_lines_image, mid_lines, (255, 0, 0))
	# show_image("lane lines", lane_lines_image)
	show_image("all lines", lane_lines_image)

	# if mid_lines is not None:
	# 	right_road, left_road = road_2_set(mask1, lane_lines, mid_lines)
	# 	# print('right co',right_road)
	# 	# print('left co',left_road)
	# 	right_road_angle = compute_steering_angle(mask1, right_road)
	# 	left_road_angle = compute_steering_angle(mask2, left_road)
	# 	# print('right',right_road_angle)
	# 	# print('left',left_road_angle)
	# 	# print(mid_lines)
	return right_road


def detect_edges(frame):

	# detect edges
	(B, G, R) = cv2.split(frame)
	B_cny = cv2.Canny(B, 190, 200)
	G_cny = cv2.Canny(G, 50, 200)
	R_cny = cv2.Canny(R, 50, 200)
	edges = cv2.Canny(B_cny, 200, 202)

	return edges


def region_of_interest(canny):
	height, width = canny.shape
	frame = np.zeros_like(canny)

	# only focus bottom half of the screen

	polygon = np.array([[
		(0, 0),
		(width, 0),
		(width, height),
		(0, height),
	]], np.int32)

	cv2.fillPoly(frame, polygon, 255)
	#show_image("frame", frame)
	masked_image = cv2.bitwise_and(canny, frame)
	return masked_image


def detect_line_segments(cropped_edges):
	# tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
	rho = 1  # precision in pixel, i.e. 1 pixel
	angle = np.pi/180  # degree in radian, i.e. 1 degree
	min_threshold = 5  # minimal of votes
	line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=3,
									maxLineGap=4)

	# if line_segments is not None:
	#     for line_segment in line_segments:
	#         logging.debug('detected line_segment:')
	#         logging.debug("%s of length %s" % (line_segment, length_of_line_segment(line_segment[0])))
	# print(type(line_segments))
	return line_segments


def detect_line_segments_norm(cropped_edges, req_lines):
	lines = cv2.HoughLines(cropped_edges, 1, np.pi/180, 50, None, 0, 0)
	if lines is not None:
		for i in range(0, len(lines)):
			rho = lines[i][0][0]
			theta = lines[i][0][1]
			a = math.cos(theta)
			b = math.sin(theta)
			x0 = a * rho
			y0 = b * rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))
			req_lines.append([[x1, y1, x2, y2]])
		req_lines_np = np.array(req_lines)
		# print(req_lines_np)
	return req_lines_np


def average_slope_intercept(frame, line_segments):
	"""
	This function combines line segments into one or two lane lines
	If all line slopes are < 0: then we only have detected left lane
	If all line slopes are > 0: then we only have detected right lane
	"""
	lane_lines = []
	if line_segments is None:
		logging.info('No line_segment segments detected')
		return lane_lines

	height, width, _ = frame.shape
	# print('height and width',height,width)
	left_fit = []
	right_fit = []

	boundary = 1/3
	# left lane line segment should be on left 2/3 of the screen
	left_region_boundary = width * (1 - boundary)
	# right lane line segment should be on left 2/3 of the screen
	right_region_boundary = width * boundary
	# print(left_region_boundary)
	for line_segment in line_segments:
		for x1, y1, x2, y2 in line_segment:
			# print(x1,y1,x2,y2)
			# cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),10)
			if x1 == x2:
				logging.info(
					'skipping vertical line segment (slope=inf): %s' % line_segment)
				continue
			fit = np.polyfit((x1, x2), (y1, y2), 1)
			# print(fit)
			slope = fit[0]
			intercept = fit[1]
			if slope < 0:
				if x1 < left_region_boundary:  # and x2 < left_region_boundary:
					left_fit.append((slope, intercept))
					# print('left',left_fit)
			else:
				if x1 > right_region_boundary:  # and x2 > right_region_boundary:
					right_fit.append((slope, intercept))
					# print('right',right_fit)

	left_fit_average = np.average(left_fit, axis=0)
	# print(left_fit_average)
	if len(left_fit) > 0:
		lane_lines.append(make_points(frame, left_fit_average))

	right_fit_average = np.average(right_fit, axis=0)
	if len(right_fit) > 0:
		lane_lines.append(make_points(frame, right_fit_average))

	# [[[316, 720, 484, 432]], [[1009, 720, 718, 432]]]
	logging.debug('lane lines: %s' % lane_lines)
	# print(lane_lines)
	return lane_lines


def average_slope_intercept_mid(frame, line_segments):
	"""
	This function combines line segments into one or two lane lines
	If all line slopes are < 0: then we only have detected left lane
	If all line slopes are > 0: then we only have detected right lane
	"""
	lane_lines = []
	if line_segments is None:
		logging.info('No line_segment segments detected')
		return lane_lines

	height, width, _ = frame.shape
	mid_fit = []

	boundary = 1/2
	# left lane line segment should be on left 2/3 of the screen
	mid_region_boundary = width * (1 - boundary)
	# print(left_region_boundary)
	for line_segment in line_segments:
		for x1, y1, x2, y2 in line_segment:
			# cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),10)
			if x1 == x2:
				logging.info(
					'skipping vertical line segment (slope=inf): %s' % line_segment)
				continue
			fit = np.polyfit((x1, x2), (y1, y2), 1)
			slope = fit[0]
			intercept = fit[1]
			if slope < 0:
				if x1 < mid_region_boundary and x2 < mid_region_boundary:
					mid_fit.append((slope, intercept))
			if slope > 0:
				if x1 > mid_region_boundary and x2 > mid_region_boundary:
					mid_fit.append((slope, intercept))
	mid_fit_average = np.average(mid_fit, axis=0)
	if len(mid_fit) > 0:
		lane_lines.append(make_points(frame, mid_fit_average))

	logging.debug('lane lines: %s' % lane_lines)  #
	# print(lane_lines)
	return lane_lines


def compute_steering_angle(frame, lane_lines):
	""" Find the steering angle based on lane line coordinate
		We assume that camera is calibrated to point to dead center
	"""
	if len(lane_lines) == 0:
		logging.info('No lane lines detected, do nothing')
		return -90

	height, width, _ = frame.shape
	if len(lane_lines) == 1:
		logging.debug(
			'Only detected one lane line, just follow it. %s' % lane_lines[0])
		x1, _, x2, _ = lane_lines[0][0]
		x_offset = x2 - x1
	else:
		_, _, left_x2, _ = lane_lines[0][0]
		_, _, right_x2, _ = lane_lines[1][0]
		# 0.0 means car pointing to center, -0.03: car is centered to left, +0.03 means car pointing to right
		camera_mid_offset_percent = 0.03
		mid = int(width / 2 * (1 + camera_mid_offset_percent))
		x_offset = (left_x2 + right_x2) / 2 - mid

	# find the steering angle, which is angle between navigation direction to end of center line
	y_offset = int(height / 2)

	# angle (in radian) to center vertical line
	angle_to_mid_radian = math.atan(x_offset / y_offset)
	# angle (in degrees) to center vertical line
	angle_to_mid_deg = int(angle_to_mid_radian * 180.0 / math.pi)
	# this is the steering angle needed by picar front wheel
	steering_angle = angle_to_mid_deg + 90

	logging.debug('new steering angle: %s' % steering_angle)
	return steering_angle


def stabilize_steering_angle(curr_steering_angle, new_steering_angle, num_of_lane_lines, max_angle_deviation_two_lines=5, max_angle_deviation_one_lane=1):
	if num_of_lane_lines == 2:
		# if both lane lines detected, then we can deviate more
		max_angle_deviation = max_angle_deviation_two_lines
	else:
		# if only one lane detected, don't deviate too much
		max_angle_deviation = max_angle_deviation_one_lane

	angle_deviation = new_steering_angle - curr_steering_angle
	if abs(angle_deviation) > max_angle_deviation:
		stabilized_steering_angle = int(curr_steering_angle
										+ max_angle_deviation * angle_deviation / abs(angle_deviation))
	else:
		stabilized_steering_angle = new_steering_angle
	logging.info('Proposed angle: %s, stabilized angle: %s' %
				 (new_steering_angle, stabilized_steering_angle))
	return stabilized_steering_angle

############################
# Utility Functions
############################


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=10):
	line_image = np.zeros_like(frame)
	if lines is not None:
		for line in lines:
			for x1, y1, x2, y2 in line:
				cv2.line(line_image, (x1, y1), (x2, y2),
						 line_color, line_width)
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
	if intercept == 0:
		intercept = 200
	if slope == 0:
		slope = -0.1
	# bound the coordinates within the frame
	x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
	x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
	return [[x1, y1, x2, y2]]


def float_to_int(x):
	if x == float('inf') or x == float('-inf'):
		return float(640)  # or a large value you choose
	return int(x)

############################
# Test Functions
############################


def test_photo(test_image):
	land_follower = HandCodedLaneFollower()
	#frame = cv2.imread(test_image)
	combo_image = land_follower.follow_lane(test_image)
	show_image('final', combo_image, True)


middle_lane = []
####################


def road_2_set(frame, lane_lines, mid_lane):
	global left_lane, right_lane
	# find middle points of two lanes
	right_road = []
	left_road = []
	print(len(lane_lines))
	if len(lane_lines) == 2:
		left_lane = lane_lines[0]
		right_lane = lane_lines[1]

		if lane_lines[0] is not None:
			left_road.append(left_lane)
		if len(mid_lane) == 1:
			left_road.append(mid_lane[0])
		if len(mid_lane) == 1:
			right_road.append(mid_lane[0])
		if lane_lines[1] is not None:
			right_road.append(right_lane)
	# print('right road',right_road)
	# print('left road',left_road)
	return right_road, left_road


def fill_hole(frame):
	im_floodfill = frame.copy()
	h, w = frame.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)
	# floodfill from point (0,0)
	cv2.floodFill(im_floodfill, mask, (0, 0), 255)
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)
	# Combine the two images to get the foreground
	im_out = frame | im_floodfill_inv
	im_out = cv2.resize(im_out, (h, w))
	# Creating kernel
	kernel = np.ones((3, 3), np.uint8)
	# Using cv2.erode() method
	image = cv2.erode(im_out, kernel)
	edges = im_out-image
	cv2.imshow('fill', edges)
	return edges
