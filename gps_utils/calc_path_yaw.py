import numpy as np

def calc_yaw(x, y):
    yaw_history = []
    for n in range(len(x)-2):
        #print(n)
        yaw = np.arctan2(y[n+1] - y[n], x[n+1] - x[n])
        #Convert rad to deg
        # yaw = np.rad2deg(yaw)
        yaw_history.append(yaw)
    # print(np.rad2deg(yaw_history))
    # print('yaw_hist',len(yaw_history))
    # print('x_lens', len(x))
    return x, y, yaw_history

