import Adafruit_PCA9685
import time
pwm = Adafruit_PCA9685.PCA9685(address=0x40, busnum=1)
pwm.set_pwm_freq(60)
pwm.set_pwm(1, 0, 350) #350 #380 #320
while True:
    pwm.set_pwm(1, 0, 380) #295
    time.sleep(1)
    pwm.set_pwm(1, 0, 350) #295
    time.sleep(1)
    pwm.set_pwm(1, 0, 410) #295
#317
# pwm.set_pwm(1, 0, 317)
