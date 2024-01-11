from itertools import count
import numpy as np
import utm
import time
import math
from sympy import Symbol, symbols, Matrix, sin, cos
from gps_utils.ego_vehicle import *
import matplotlib.pyplot as plt

# Kalman filter updates with every new measurments.
x0, x1, x2, x3 = [], [], [], [] #4 state for prediction
Zx, Zy = [], [] # measurement parameter
Px, Py, Pdx, Pdy = [], [], [], [] # error covarianace of 4 state
Kx, Ky, Kdx, Kdy = [], [], [], [] # Kalman gain 
numstates=4 # number of elements in state vector
i=0 # count number of updates
cx_1=0
cy_1=0
cx=0
cy=0
dt=1/20 #s
# Initiate coordinate
x=np.matrix([[0.0, 0.0, 0.0, 0.0]],dtype='float').T

def l2ar(list):
    arr=np.array(list,dtype='float')
    return arr

def latlong2XY(lat,lng,arc,lat_1,lng_1):
    dx = arc * np.cos(lat*np.pi/180.0) * (lng-lng_1) # in m
    dy = arc * (lat-lat_1) # in m
    return dx,dy

def latlngToUTM(lat, lng):
    pos = utm.from_latlon(lat, lng)
    # print(pos[0], pos[1])
    return pos

def Global2local(latitude,longitude,altitude,lat_1,lng_1,alt_1):
    latitude=l2ar(latitude)
    longitude=l2ar(longitude)
    altitude=l2ar(altitude)
    RadiusEarth = 6378388.0 # m
    # print(altitude)
    arc= 2.0*np.pi*(RadiusEarth+altitude)/360.0 # m/°
    # dx,dy=latlong2XY(latitude,longitude,arc,lat_1,lng_1)
    pos = latlngToUTM(latitude, longitude)
    # return dx,dy
    return pos[0], pos[1]

def sqr_error(cx_1,cy_1,cx,cy):
    error= np.sqrt(cx**2+cy**2) - np.sqrt(cx_1**2+cy_1**2)
    if error != 0:
        return True

def savestates(x, Z, P, K):
    x0.append(float(x[0]))
    x1.append(float(x[1]))
    x2.append(float(x[2]))
    x3.append(float(x[3]))
    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))    
    Px.append(float(P[0,0]))
    Py.append(float(P[1,1]))
    Pdx.append(float(P[2,2]))
    Pdy.append(float(P[3,3]))
    Kx.append(float(K[0,0]))
    Ky.append(float(K[1,0]))
    Kdx.append(float(K[2,0]))
    Kdy.append(float(K[3,0]))
    return Px,Py,Pdx,Pdy


def run_rlt(GPS,raw_yaw,id,i):
    global x0, x1, x2, x3, x
    global Zx, Zy
    global Px, Py, Pdx, Pdy
    global Kx, Ky, Kdx, Kdy
    global numstates
    # global i
    global cx_1, cy_1, cx, cy
    global lat_1, lng_1, alt_1
    # mx,my=Global2local(GPS_latitude,GPS_longitude,GPS_altitude,lat_1,lng_1,alt_1)
    if (i==0):
        mx,my,yaw=vehicle(GPS,raw_yaw,id)
        x = np.matrix([[mx, my, 0.5*np.pi, 0.0]],dtype='float').T
    if (i>0): #read more than 1 time
        begin_time=time.time()
        cx, cy, yaw = vehicle(GPS,raw_yaw,id)
        vs, psis, dts, xs, ys, lats, lons = symbols('v \psi T x y lat lon')
        # Dynamic equation of car model
        gs = Matrix([[xs+vs*dts*cos(psis)],
                [ys+vs*dts*sin(psis)],
                [psis],
                [vs]])
        state = Matrix([xs,ys,psis,vs])
        jacob=gs.jacobian(state)# the purpose of jacobian is to estimate working point of nonlinear system model 
        # print(jacob)
        P = np.eye(numstates)*500.0# initiate covariance matrix
        # Provide offset for course
        hs = Matrix([[xs],
                    [ys]])
        JHs=hs.jacobian(state)
        varGPS = 4.0 # Standard Deviation of GPS Measurement-apply this deviation for both x and y axis
        R = np.diag([varGPS**2.0, varGPS**2.0])
        # identity matrix
        I = np.eye(numstates)
        # the beginning yaw rate start at 45 degree 
        dt=(time.time()-begin_time)*2
        x[0] = x[0] + dt*x[3]*np.cos(x[2])
        x[1] = x[1] + dt*x[3]*np.sin(x[2])
        x[2] = (x[2]+ np.pi) % (2.0*np.pi) - np.pi
        x[3] = x[3]
        a13 = -dt*x[3]*np.sin(x[2])
        a14 = dt*np.cos(x[2])
        a23 = dt*x[3]*np.cos(x[2])
        a24 = dt*np.sin(x[2])
        JA = np.matrix([[1.0, 0.0, a13, a14],
                        [0.0, 1.0, a23, a24],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]],dtype='float')
        # Calculate the Process Noise Covariance Matrix
        sGPS     = 0.5*9*dt**2  # assume 8.8m/s2 as maximum acceleration
        sCourse  = 1.0*dt # assume 0.5rad/s as maximum turn rate
        sVelocity= 3*dt # assume 8.8m/s2 as maximum acceleration
        Q = np.diag([sGPS**2, sGPS**2, sCourse**2, sVelocity**2])
        
        # Project the error covariance ahead
        P = JA*P*JA.T + Q
        # print(P)
        # Measurement Update (Correction)
        # ===============================
        # Measurement Function
        hx = np.matrix([[float(x[0])],
                        [float(x[1])]],dtype='float')
        if (sqr_error(cx_1,cy_1,cx,cy) == True): # 
            JH = np.matrix([[1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0]],dtype='float')
        else: # every other step
            JH = np.matrix([[0.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 0.0]],dtype='float')        

        # S = JH*P*JH.T + R# measurement model
        K = (P*JH.T) * np.linalg.inv(JH*P*JH.T + R)
        # print(K)
        # Update the estimate via
        Z = np.matrix([[cx,cy]],dtype='float').T
        # Innovation or Residual
        y = Z - (hx)
        # print(y)                         
        x = x + (K*y)
        # Update the error covariance
        P = (I - (K*JH))*P
        # Save states for Plotting
        Px,Py,Pdx,Pdy=savestates(x, Z, P, K)
        # print(x[0],x[1])
    # i=i+1
    cx_1=cx
    cy_1=cy
    # if (i>6):
    #     plt.plot(x0[-5],x1[-5],label='EKF',c='r')
    #     plt.xlabel('X [m]')
    #     plt.ylabel('Y [m]')
    #     plt.title('Position')
    #     plt.legend(loc='best'
    #     plt.axis('equal')
    #     plt.show()
    return x.item(0),x.item(1)

