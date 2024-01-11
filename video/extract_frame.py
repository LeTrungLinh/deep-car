# Importing all necessary libraries
import cv2
import os
  
# Read the video from specified path
cam = cv2.VideoCapture("video/vid4.mp4")
  
try:
      
    # creating a folder named data
    if not os.path.exists('data'):
        os.makedirs('data')
  
# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')
  
# frame
i=0
currentframe = 0
  
while(True):
      
    # reading from frame
    ret,frame = cam.read()
  
    if ret:
        # if video is still left continue creating images
        if (i%3)==0:
            name = 'image_7_3_2022/' + str(currentframe) + '.jpg'
            print ('Creating...' + name)
            currentframe += 1
            cv2.imwrite(name, frame)
  
        # writing the extracted images
        
        i=i+1
        # increasing counter so that it will
        # show how many frames are created
    
    else:
        break
  
# Release all space and windows once done
cam.release()
cv2.destroyAllWindows()