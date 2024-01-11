import Jetson.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
mode = GPIO.getmode()
while(True):
    GPIO.setup(11, GPIO.IN)
    state = GPIO.input(11)
    print(state)