import cv2
import numpy as np
import logging
import math
import datetime
import sys

def detect_edge(frame):# detect canny edge
    edge=cv2.Canny(frame,100,400)
    # kernel = np.ones((5,5), np.uint8)
    # # dilate to open pixel
    # edge_dilation = cv2.dilate(edge, kernel, iterations=1)
    return edge

def region_of_interest_mid(edge):# crop image
    height, width = edge.shape
    frame = np.zeros_like(edge)

    # only focus bottom half of the screen

    polygon = np.array([[
        (0, 0),
        (width/2, 0),
        (width/2, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(frame, polygon, 255)
    #show_image("frame", frame)
    masked_image = cv2.bitwise_and(edge, frame)
    return masked_image

def detect_mid_lane(cropped_edges):# use HoughLinesP to detect mid line
    #tuning min_threshold, minLineLength, maxLineGap is a trial and error process by hand
    rho = 1  # precision in pixel, i.e. 1 pixel
    angle = np.pi / 180  # degree in radian, i.e. 1 degree
    min_threshold = 1  # minimal of votes
    line_segments = cv2.HoughLinesP(cropped_edges, rho, angle, min_threshold, np.array([]), minLineLength=2,
                                    maxLineGap=6)
    return line_segments

def display_lines(frame, lines, line_color=(0, 0, 255), line_width=7):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image

def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height  # bottom of the frame
    y2 = int(y1 * 1/6)  # make points from middle of the frame down

    # bound the coordinates within the frame
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]

def fit_line(frame, line_segments):
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
    mid_region_boundary = width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    # print(left_region_boundary)
    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            # cv2.line(frame,(x1,y1),(x2,y2),(255,0,0),10)
            if x1 == x2:
                logging.info('skipping vertical line segment (slope=inf): %s' % line_segment)
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < mid_region_boundary and x2 < mid_region_boundary:
                    mid_fit.append((slope, intercept))
    mid_fit_average = np.average(mid_fit, axis=0)
    if len(mid_fit) > 0:
        lane_lines.append(make_points(frame, mid_fit_average))

    logging.debug('lane lines: %s' % lane_lines)  # 
    # print(lane_lines)
    return lane_lines

def mid_lane(frame):# detect mid lane
    edges = detect_edge(frame)

    cropped_edges = region_of_interest_mid(edges)
    kernel = np.ones((5,5),np.uint8)
    dilate_line = cv2.dilate(cropped_edges, kernel, iterations=1)

    mid_lane = detect_mid_lane(cropped_edges)

    line_segment_image = display_lines(frame, mid_lane)
    

    straight = fit_line(frame, mid_lane)
    straight_lines_image = display_lines(frame, straight)

    cv2.imshow('shit',straight_lines_image)
    cv2.imshow('shit2',cropped_edges)
    return straight



