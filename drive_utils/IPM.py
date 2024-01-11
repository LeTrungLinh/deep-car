import cv2
import numpy as np

def compute_perspective_transform(corner_points,w,h,image):
    corner_points_array = np.float32(corner_points)
    img_params = np.float32([[w/2.1,h],[w-w/2.1,h],[0,0],[w,0]])
    matrix = cv2.getPerspectiveTransform(corner_points_array, img_params)
    img_transformed = cv2.warpPerspective(image, matrix, (w,h))
    return matrix, img_transformed

def compute_point_perspective_transform(matrix,list_downoids):
    list_points = np.float32(list_downoids).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(list_points, matrix)
    transformed_points_list = list()
    try:
        for i in range(0, transformed_points.shape[0]):
            transformed_points_list.append([transformed_points[i][0][0], transformed_points[i][0][1]])
    except AttributeError:
        pass
    return transformed_points_list