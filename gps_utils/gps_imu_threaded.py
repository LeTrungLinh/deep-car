import os
import multiprocessing as mp
import threading
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rand
import socket, traceback, select

from matplotlib.animation import FuncAnimation
from gps_utils.draw_car import Description
from gps_utils.calc_path_yaw import calc_yaw
from gps_utils.stanley import StanleyController
from gps_utils.kinematic_model import KinematicBicycleModel
from gps_utils.ego_vehicle import *

class Simulation:
    def __init__(self):
        fps = 50
        self.dt = 1/fps
        self.map_size = 60
        self.frames = 2000
        self.loop = False

class Path:
    def __init__(self):
        dir_path = 'routes_data/map.csv'
        df = pd.read_csv(dir_path, encoding='utf-8')

        x = df['X-axis'].values
        y = df['Y-axis'].values
        # ds = 0.1

        self.px, self.py, self.pyaw = calc_yaw(x, y)

class Car:
    def __init__(self, init_x, init_y, init_yaw, px, py, pyaw, dt):

        #Model params
        self.x = init_x
        self.y = init_y
        self.yaw = init_yaw
        self.v = 0.0
        self.delta = 0
        self.L = 2.5
        self.max_steer = np.deg2rad(33)
        self.dt = dt
        self.c_r = 0.01
        self.c_a = 2.0

        #Tracker params
        self.px = px
        self.py = py
        self.pyaw = pyaw
        self.k = 8.0
        self.ksoft = 1.0 #?Stanley has ksoft and k
        self.kyaw = 0.01
        self.ksteer = 0.0
        self.crosstrack_error = None
        self.target_id = None #??

        #Car discription
        self.length = 4.5 # [m] car length
        self.width = 2.0
        self.rear2wheel = 1.0
        self.wheel_dia = 0.15 * 2
        self.wheel_width = 0.2
        self.tread = 0.7
        self.color = 'black'

        self.tracker = StanleyController(self.k, self.ksoft, self.kyaw, self.ksteer, self.max_steer, self.L, self.px, self.py, self.pyaw)
        self.kbm = KinematicBicycleModel(self.L, self.max_steer, self.dt, self.c_r, self.c_a)

    def drive(self,lock):
        throttle = 250
        self.delta, self.target_id, self.crosstrack_error = self.tracker.stanley_control(self.x, self.y, self.yaw, self.v, self.delta)

        #---------LẤY GIÁ TRỊ CẢM BIẾN VÀ XỬ LÍ TẠI ĐÂY---------#
        
        raw_yaw = get_orientation(lock)[0]
        gps = get_gps(lock)
        
        self.x, self.y, self.yaw = vehicle(gps,raw_yaw,self.pyaw)
        self.v = 20 
        # _,_,self.yaw, self.v, _ = self.kbm.kinematic_model(self.x, self.y, self.yaw, self.v, throttle, self.delta)

        print('yaw', self.yaw)
        print('delta',self.delta)

        # Print params
        #os.system('cls' if os.name=='nt' else 'clear')
        print(f"Cross-track term: {self.crosstrack_error}")
        print('throt', throttle)
        print('v',self.v)

def get_orientation(lock):
    '''
    Safely get orientation from the temporary file.
    
    Parameters
    ----------
    lock : mp.Lock
        Lock used for synchronising access to the temporary file.
    Returns
    -------
    orientation : np.array of floats
        The orientation of the phone in alpha, beta, gamma.
    '''
    # Read the orientation data. Ensure the other thread has not opened the 
    # file at the same time
    global orientation
    lock.acquire()    
    with open('sensor_log/yaw.txt', 'rb') as file:
        file.seek(0)
        data = file.read().decode()
    lock.release()

    # Find and read in the orientation data
    entries = np.array(data.split(',')).astype(float)
    index = np.squeeze(np.where(entries == 81))
    orientation = np.array(entries[index + 1 : index + 4])
    return orientation

def get_gps(lock):
    '''
    Safely get gps from the temporary file.
    
    Parameters
    ----------
    lock : mp.Lock
        Lock used for synchronising access to the temporary file.
    Returns
    -------
    gps : np.array of floats
        The gps position of the phone.
    '''
    # Read the GPS data. Ensure the other thread has not opened the 
    # file at the same time
    global gps
    lock.acquire()    
    with open('sensor_log/gps.txt', 'rb') as file:
        file.seek(0)
        data = file.read().decode()
    lock.release()

    # Find and read in the orientation data
    entries = np.array(data.split(',')).astype(float)
    index = np.squeeze(np.where(entries == 1))
    # try: 
    gps = np.array(entries[2 : 4])
    return gps

# SIMULATE AND DRAW CAR THEN PLOT ON MAP 
def simulation(lock):
    sim = Simulation()
    path = Path()
    car = Car(path.px[0], path.py[0], path.pyaw[0] ,path.px, path.py, path.pyaw, sim.dt)
    desc = Description(car.length, car.width, car.rear2wheel, car.wheel_dia, car.wheel_width, car.tread, car.L)
    
    interval = sim.dt * 10#**3

    fig, ax = plt.subplots(1,2,figsize=(10, 5))
    # PLOT CURRENT LOCATION OF VEHICLE - LOCAL FRAME
    ax[0].set_aspect('equal')
    # Plot route
    ax[0].plot(path.px, path.py, '--', color='blue')
    #ax[0].scatter(path.px,path.py)
    annotation = ax[0].annotate(f'{car.x:.1f}, {car.y:.1f}', xy=(car.x, car.y + 5), color='black', annotation_clip=False)
    target, = ax[0].plot([], [], '+r')
    outline, = ax[0].plot([], [], color=car.color)
    fr, = ax[0].plot([], [], color=car.color)
    rr, = ax[0].plot([], [], color=car.color)
    fl, = ax[0].plot([], [], color=car.color)
    rl, = ax[0].plot([], [], color=car.color)
    rear_axle, = ax[0].plot(car.x, car.y, '+', color=car.color, markersize=2)
    ax[0].grid()
    # PLOT MAP AND VEHICLE - GLOBAL FRAME
    ax[1].set_aspect('equal')
    # Plot route
    ax[1].plot(path.px, path.py, '--', color='blue') 
    # Plot vehicle
    rear_axle2, = ax[1].plot(car.x, car.y, color='red', marker='v', markersize=10)
    ax[1].grid()

    def animate(frame):
        # Camera tracks car
        ax[0].set_xlim(car.x - sim.map_size + 40, car.x + sim.map_size - 40)
        ax[0].set_ylim(car.y - sim.map_size + 40, car.y + sim.map_size - 40)
        
        # Drive and draw car - local view
        car.drive(lock)
        outline_plot, fr_plot, rr_plot, fl_plot, rl_plot = desc.plot_car(car.x, car.y, car.yaw, car.delta)
        outline.set_data(outline_plot[0], outline_plot[1])
        fr.set_data(fr_plot[0], fr_plot[1])
        rr.set_data(rr_plot[0], rr_plot[1])
        fl.set_data(fl_plot[0], fl_plot[1])
        rl.set_data(rl_plot[0], rl_plot[1])
        rear_axle.set_data(car.x, car.y)

        # Show car's target
        target.set_data(path.px[car.target_id], path.py[car.target_id])
        # Annotate car's coordinate above car
        annotation.set_text(f'{car.x:.1f}, {car.y:.1f}')
        annotation.set_position((car.x, car.y + 2))

        # Draw map and car - map view
        # Drive and draw car
        rear_axle2.set_data(car.x, car.y+5)

        # Show car's target
        target.set_data(path.px[car.target_id], path.py[car.target_id])

        # Annotate car's coordinate above car
        annotation.set_text(f'{car.x:.1f}, {car.y:.1f}')
        annotation.set_position((car.x, car.y + 5))

        plt.title(f'{sim.dt*frame:.2f}s')
        plt.xlabel(f'Speed: {car.v:.2f} m/s')

        return outline, fr, rr, fl, rl, rear_axle, target,
        

    _ = FuncAnimation(fig, animate, frames=sim.frames, interval=100, repeat=sim.loop)
    # anim.save('animation.gif', writer='imagemagick', fps=50)
    plt.show()

def client(lock, file_name):
    '''
    Reads a datastream from a socket. The data is sent by the "IMU + GPS 
    Stream App" for android. Writes the most recent orientation data to a file
    and writes all data into a csv file.
    
    Parameters
    ----------
    lock : mp.Lock
        Pass in a lock for synchronisation.
    file_name : string
        The file name for logging data
    Returns
    -------
    None.
    '''
    
    # Lookup data for each sensor reading. 'number' is the integer identifier
    # sent by the app before the data, 'n_col' is the number of readings for
    # the sensor (e.g. x, y, and z readings for accelerometer)
    lookup = [
        {'name' : 'GPS', 'number' : 1, 'n_col' : 3},
        {'name' : 'Accelerometer', 'number' : 3, 'n_col' : 3},
        {'name' : 'GyroScope', 'number' : 4, 'n_col' : 3},
        {'name' : 'Magnetic Field', 'number' : 5, 'n_col' : 3},
        {'name' : 'Orientation', 'number' : 81, 'n_col' : 3}
        ]

    numbers = [sensor['number'] for sensor in lookup]
    
    # Create the header for writing to the logging file
    column_name_line = 'Time,'
    for sensor in lookup:
        for i in range(sensor['n_col']):
            column_name_line += sensor['name'] + ' ' + str(i) + ','
    column_name_line += '\n'
    
    
    # Open the csv file with write only permission
    with open(file_name, 'wb') as file:
        # Write the header to the csv
        file.write(column_name_line.encode())
        
        
        # Socket information (port needs to match port used in the app)
        host = ''
        port = 5555
        
        # Set up client
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            s.bind((host, port))

            print('Waiting to recieve data...')
            
            # Process the stream...
            read_write(s, file, numbers, 10, lock) 
            while(1):
                read_write(s, file, numbers, 1, lock)

def read_write(s, write_file, numbers, timeout, lock):
    try:
        start_time = time.time()
        while(1):
            # Make sure recvfrom does not block other threads
            ready = select.select([s], [], [], 0.001)
            if ready[0]:
                message, address = s.recvfrom(1024)
                break
            else:
                time.sleep(0.001)
                if time.time() - start_time >= timeout:
                    print('Timeout')
                    raise SystemExit
            

        write_string = ''
        rest = message.decode() 

        while(1):
            loc = rest.find(',')
            if loc == -1:
                write_string += rest +'\n'
                break
            else:
                seg = rest[:loc]
                num = float(seg)
                rest = rest[loc + 1 :]
                
            if not num in numbers:
                write_string += seg + ','

            if num == 81:
                lock.acquire()
                with open('sensor_log/yaw.txt', 'wb') as temp_file:
                    temp_file.write(message)   
                lock.release()

            elif num == 1:
                lock.acquire()
                with open('sensor_log/gps.txt', 'wb') as temp_file:
                    temp_file.write(message)   
                lock.release()
                    
        message_line = (write_string).encode()
        write_file.write(message_line)

        #print (message)
    
    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        traceback.print_exc()

if __name__ == '__main__':
    lock = threading.Lock()  
    file_name = 'sensor_log/log_file.csv'
    
    client_process = threading.Thread(target = client, args = (lock,file_name))
    animation_process = threading.Thread(target = simulation, args = (lock,))

    animation_process.start()
    client_process.start()

    client_process.join()

    # animation_process.terminate() 
    # vision_process.terminate()
