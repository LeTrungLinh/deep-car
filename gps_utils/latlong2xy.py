import math
import numpy as np
import geopy.distance
import utm

radius = 6371    #Earth Radius in KM

class referencePoint:
    def __init__(self, scrX, scrY, lat, lng):
        self.scrX = scrX
        self.scrY = scrY
        self.lat = lat
        self.lng = lng

# Specify 2 coordinates for convert
# c1 = (10.8360721, 106.7766884)
# c2 = (10.8335676, 106.7790829) #HOME
c1 = (10.849575, 106.7711649)
c2 = (10.8538745, 106.7743475)

# Calculate global X and Y for top-left reference point        
# p0 = referencePoint(0, 0, 10.84995, 106.77113)
# # Calculate global X and Y for bottom-right reference point
# p1 = referencePoint(334.5836207775342, 428.0748040340691, 10.8538, 106.77419) 
# p1 = referencePoint(500, 500, 10.84995, 106.77419) 

# Calculate X and Y distance between 2 coord points
dx = geopy.distance.geodesic(c1, (c1[0], c2[1])).m
dy = geopy.distance.geodesic(c1, (c2[0], c1[1])).m
# print('dx,dy', dx, dy)
# Calculate global X and Y for top-left reference point        
p0 = referencePoint(0, 0, c1[0], c1[1])
# Calculate global X and Y for bottom-right reference point
p1 = referencePoint(dx, dy, c2[0], c2[1]) 
# p1 = referencePoint(500, 500, 10.84995, 106.77419) 

# This function converts lat and lng coordinates to GLOBAL X and Y positions
def latlngToGlobalXY(lat, lng):
    # Calculates x based on cos of average of the latitudes
    x = radius*lng*math.cos(np.radians(p0.lat + p1.lat)/2)
    # Calculates y based on latitude
    y = radius*lat
    return {'x': x, 'y': y}

# Calculate global X and Y for top-left reference point
p0.pos = latlngToGlobalXY(p0.lat, p0.lng)
# Calculate global X and Y for bottom-right reference point
p1.pos = latlngToGlobalXY(p1.lat, p1.lng)

# This function converts lat and lng coordinates to SCREEN X and Y positions
def latlngToScreenXY(lat, lng):
    # Calculate global X and Y for projection point
    pos = latlngToGlobalXY(lat, lng)
    # Calculate the percentage of Global X position in relation to total global width
    perX = ((pos['x']-p0.pos['x'])/(p1.pos['x'] - p0.pos['x']))
    # Calculate the percentage of Global Y position in relation to total global height
    perY = ((pos['y']-p0.pos['y'])/(p1.pos['y'] - p0.pos['y']))

    # Returns the screen position based on reference points
    return np.array([p0.scrX + (p1.scrX - p0.scrX)*perX,
    p0.scrY + (p1.scrY - p0.scrY)*perY], dtype=np.float32)
    
def latlngToUTM(lat, lng):
    pos = utm.from_latlon(lat, lng)
    # print(pos[0], pos[1])
    return pos




