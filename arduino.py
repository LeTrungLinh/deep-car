import serial
import numpy as np
import Adafruit_PCA9685
import time
from numpy import interp

pwm = Adafruit_PCA9685.PCA9685(address=0x40, busnum=1)
pwm.set_pwm_freq(60)
pwm.set_pwm(1, 0, 350) #350 #380 #320
arduino = serial.Serial('/dev/ttyUSB0', timeout=1, baudrate=9600)

while True:
    s = str(arduino.readline())[2:31]
    # print(s)
    try:
        mode1 = float(s[0:4])
        mode2 = float(s[5:9])
        throth = float(s[15:19])
        servo = float(s[20:24])
        # print((throth),(servo),(mode1),(mode2))
        aglservo = interp(servo,[1100,1900],[310,490])
        throth = interp(throth,[1100,1900],[250,450])
        print(throth)
        pwm.set_pwm(0, 0, int(aglservo)) #295
        pwm.set_pwm(1, 0, int(throth)) #295
        #time.sleep(0.2)
    except ValueError:
        pass
