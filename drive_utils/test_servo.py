import Adafruit_PCA9685
import time
pwm = Adafruit_PCA9685.PCA9685(address=0x40, busnum=1)
pwm.set_pwm_freq(80)
pwm.set_pwm(0, 0, 380) #295
while True:
    pwm.set_pwm(1, 0, 260) #295
    time.sleep(1)
    pwm.set_pwm(1, 0, 500) #295
    time.sleep(1)
#317
# pwm.set_pwm(1, 0, 317)