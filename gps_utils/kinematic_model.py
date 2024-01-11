import numpy as np
import math as m

normalise_angle = lambda angle : m.atan2(m.sin(angle), m.cos(angle))


"""
2D Kinematic Bicycle Model

At initialisation
:param L:           (float) vehicle's wheelbase [m]
:param max_steer:   (float) vehicle's steering limits [rad]
:param dt:          (float) discrete time period [s]
:param c_r:         (float) vehicle's coefficient of resistance 
:param c_a:         (float) vehicle's aerodynamic coefficient

At every time step
:param x:           (float) vehicle's x-coordinate [m]
:param y:           (float) vehicle's y-coordinate [m]
:param yaw:         (float) vehicle's heading [rad]
:param v:           (float) vehicle's velocity in the x-axis [m/s]
:param throttle:    (float) vehicle's accleration [m/s^2]
:param delta:       (float) vehicle's steering angle [rad]

:return x:          (float) vehicle's x-coordinate [m]
:return y:          (float) vehicle's y-coordinate [m]
:return yaw:        (float) vehicle's heading [rad]
:return v:          (float) vehicle's velocity in the x-axis [m/s]
:return delta:      (float) vehicle's steering angle [rad]
:return omega:      (float) vehicle's angular velocity [rad/s]
"""

class KinematicBicycleModel():

    def __init__(self, L=1.0, max_steer=0.7, dt=0.05, c_r=0.0, c_a=0.0):

        self.dt = dt
        self.L = L
        self.max_steer = max_steer
        self.c_r = c_r
        self.c_a = c_a

    def kinematic_model(self, x, y, yaw, v, throttle, delta):

        # Compute velocity 
        f_load = v * (self.c_r + self.c_a * v)
        v = v + (throttle - f_load) * self.dt

        # Clip delta value to max_steer
        delta = np.clip(delta, -self.max_steer, self.max_steer)

        # Compute state in discrete time model
        x = x + v * np.cos(yaw) * self.dt
        y = y + v * np.sin(yaw) * self.dt
        yaw = yaw + (v * np.tan(delta) / self.L) * self.dt
        yaw = normalise_angle(yaw)

        return x, y, yaw, v, delta



