import numpy as np
from gps_utils.latlong2xy import *

def vehicle(gps,raw_yaw,id):
    global x,y,yaw
    lat,lon = gps[0],gps[1]
    # print('heree',lat,lon,id)
    pos = latlngToUTM(lat,lon)
    # if 0<pos[0]<500:
    x = pos[0]
    y = pos[1] 
    # x = 693652
    # y = 1200111
    # else: x,y = 0,0
    # print('raw_yaw', raw_yaw)
    yaw = np.deg2rad(-raw_yaw+90+10) # -20
    # print('yaw_rad',yaw)
    # print('yaw', -raw_yaw+90)
    # print(lat,lon,zaxis)
    # print('coord',x,y,-raw_yaw+90)
    return x,y,yaw

