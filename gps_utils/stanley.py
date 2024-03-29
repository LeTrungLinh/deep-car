import numpy as np
import math as m
from numpy.core.numeric import cross

normalise_angle = lambda angle : m.atan2(m.sin(angle), m.cos(angle))

"""
Stanley Controller

At initialisation
:param control_gain:                (float) time constant [1/s]
:param softening_gain:              (float) softening gain [m/s]
:param yaw_rate_gain:               (float) yaw rate gain [rad]
:param steering_damp_gain:          (float) steering damp gain
:param max_steer:                   (float) vehicle's steering limits [rad]
:param wheelbase:                   (float) vehicle's wheelbase [m]
:param path_x:                      (numpy.ndarray) list of x-coordinates along the path
:param path_y:                      (numpy.ndarray) list of y-coordinates along the path
:param path_yaw:                    (numpy.ndarray) list of discrete yaw values along the path
:param dt:                          (float) discrete time period [s]

At every time step
:param x:                           (float) vehicle's x-coordinate [m]
:param y:                           (float) vehicle's y-coordinate [m]
:param yaw:                         (float) vehicle's heading [rad]
:param target_velocity:             (float) vehicle's velocity [m/s]
:param steering_angle:              (float) vehicle's steering angle [rad]

:return limited_steering_angle:     (float) steering angle after imposing steering limits [rad]
:return target_index:               (int) closest path index
:return crosstrack_error:           (float) distance from closest path index [m]
"""
class StanleyController:
    
    def __init__(self, control_gain=2.5, softening_gain=1.0, yaw_rate_gain=0.0, steering_damp_gain=0.0, max_steer=np.deg2rad(24), wheelbase=0.0, path_x=None, path_y=None, path_yaw=None):

        self.k = control_gain
        self.k_soft = softening_gain
        self.k_yaw_rate = yaw_rate_gain
        self.k_damp_steer = steering_damp_gain
        self.max_steer = max_steer
        self.L = wheelbase

        self.px = path_x
        self.py = path_y
        self.pyaw = path_yaw

        self.yaw_hist = 0

    def find_target_path_id(self, x, y, yaw):

        # Position of front axle
        fx = x + self.L * np.cos(yaw) 
        fy = y + self.L * np.sin(yaw)

        dx = fx - self.px # Find x-axis of front axle relative to the path
        dy = fy - self.py # Find y-axis of front axle relative to the path

        d = np.hypot(dx, dy) # Find the distance from the front axle to the path
        target_index = np.argmin(d) # Find the shortest distance in the array
        # print('index', target_index)

        return target_index, dx[target_index], dy[target_index], d[target_index]

    def calculate_yaw(self, target_index, yaw):
        # print('index',target_index) 
        try:
            yaw_error = normalise_angle(self.pyaw[target_index] - yaw)
            self.yaw_hist = yaw_error
        # print(yaw_error)
        except Exception as e:
            yaw_error = self.yaw_hist
            print("type error: " + str(e))
            pass
        
        return yaw_error

    def calculate_cross_track_error(self, target_velocity, yaw, dx, dy, absolute_error):

        front_axle_vector = [np.sin(yaw), -np.cos(yaw)]
        nearest_path_vector = [dx, dy]
        crosstrack_error = np.sign(np.dot(nearest_path_vector, front_axle_vector)) * absolute_error
        crosstrack_steering_error = np.arctan2((self.k * crosstrack_error), (self.k_soft + target_velocity))

        return crosstrack_steering_error, crosstrack_error

    def calculate_yaw_rate(self, target_veloocity, steering_angle):

        yaw_rate_error = self.k_yaw_rate * (-target_veloocity * np.sin(steering_angle)) / self.L

        return yaw_rate_error

    def calculate_steering_delay(self, computed_steering_angle, previous_steering_angle):

        steering_delay_error = self.k_damp_steer * (computed_steering_angle - previous_steering_angle)

        return steering_delay_error

    def stanley_control(self, x, y, yaw, target_velocity, steering_angle):

        target_index, dx, dy, absolute_error = self.find_target_path_id(x, y, yaw)
        yaw_error = self.calculate_yaw(target_index, yaw)
        crosstrack_steering_error, crosstrack_error = self.calculate_cross_track_error(target_velocity, yaw, dx, dy, absolute_error)
        yaw_rate_damping = self.calculate_yaw_rate(target_velocity, steering_angle)

        # desired_steering_angle = yaw_error + crosstrack_steering_error + yaw_rate_damping
        desired_steering_angle = yaw_error + crosstrack_steering_error*0.25 + yaw_rate_damping

        # Constrains steering angle to vehicle limits
        desired_steering_angle += self.calculate_steering_delay(desired_steering_angle, steering_angle)
        limited_steering_angle = np.clip(desired_steering_angle, -self.max_steer, self.max_steer)

        return limited_steering_angle, target_index, crosstrack_error

         
