import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os


def process_img (image):
    h, w = image.shape[:2]
    


    img_smoothed = cv2.medianBlur(image,7)
    img_hsv = cv2.cvtColor(img_smoothed, cv2.COLOR_BGR2HSV)
    #print(w)
    # ImShow(img_smoothed,title="Smoothed Image")
    # ImShow(img_hsv,title="HSV color space Image")
    # Calculate histogram
    

    counts, bins = np.histogram(img_hsv[int(h*0.5):h, :, 1])

    value = bins[list(counts).index(max(counts))]
    if value == 0:
        value = 10
    a=(int(190*0.5),290)
    mask = np.ones(a)*255
    mask[img_hsv[int(h*0.5):h, :,1] < value - 15] = 0
    mask[img_hsv[int(h*0.5):h, :, 1] > value + 25] = 0

    img_bw = np.zeros(mask.shape)
    img_bw[np.where(mask == 255)] = 255


    img_bw = img_bw.astype(np.uint8)

    contours,_ = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
    contours = sorted(contours, key = cv2.contourArea, reverse=True)[:1]
    line_segmented = np.zeros(img_bw.shape)
    cv2.drawContours(line_segmented, contours[0], -1, (255), 3)
    # Draw mask of Lane into Original Image 
    mask = np.zeros(img_bw.shape)
    cv2.drawContours(mask, [contours[0]], -1, (255), -1)
    image[int(h*0.5):h, :, 0] = np.where(mask == 255 ,255, image[int(h*0.5):h, :, 0])

    # ImShow(image,title="Visualize Original Image")
    # ImShow(mask,title="Mask of Lane" ,cmap="gray")
    # draw polygon
    rows, cols = (190*0.5,290)
    bottom_left  = [0, rows-10]
    top_left     = [0+10, rows*0.1]
    bottom_right = [cols, rows-10]
    top_right    = [cols-10, rows*0.1] 
    polygon = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    mask = np.zeros_like(line_segmented)
    mask = cv2.fillPoly(mask, polygon, (255))    
    line_segmented = cv2.bitwise_and(line_segmented, mask)

    # ImShow(mask,title="Region of interest" ,cmap="gray")
    # ImShow(line_segmented,title="line Detection" ,cmap="gray")
    line_segmented = line_segmented.astype(np.uint8)

    return line_segmented
def line_detect(imge, threshold,lower_angle_lim, upper_angle_lim ):
    '''
    line_detect(imge, threshold,lower_angle_lim, upper_angle_lim ) function is used to detect lines from the processed image which have a houghscore greater than the threshold and whose angles does not lie between the lower and upper angle limit.(this is to ensure that the lines are almost vertical)

    input: 	imge - numpy.ndarray, the input image
        threshold - integer, the min threshold values which th hough score of detected lines should satisfy
        lower_angle_lim and upper_angle_lim - floats, the detected lines would not be between these angle limits.
    
    output:
        req_lines - list, a list of all lines which satisfy the required conditions

    '''
    all_lines = cv2.HoughLines(imge,1,np.pi/180,threshold)
    req_lines = []
    if isinstance(all_lines,np.ndarray):
        #ensuring that all the detected lines are nearly vertical. i.e. not lying between the two angle limits
        for each_line in all_lines:
            for rho,theta in each_line:
                if theta < np.pi*lower_angle_lim/180 or theta> np.pi*upper_angle_lim/180:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    req_lines.append([x1,y1,x2,y2,rho,theta])
    return req_lines


#function which drawsthe detected lines on to an image
def draw_line(imge,drw_lines,colour):
    '''
    draw_line(imge,drw_lines) function is used to draw lines in an image

    input: imge - numpy.ndarray, the image on which the lines has to be drawn
        drw_lines - list, the list of rho and theta values of the line, usually obtained from houghlines
        colour - tuple, tuple of RGB values of the colour inwhich the lines have to be drawn

    output: imge - numpy.ndarray, the output image with the lines drawn

    '''
    for x1,y1,x2,y2,rho,theta in drw_lines:
            cv2.line(imge,(x1,y1),(x2,y2),colour,2)
    
    return imge

#function to determine intercept of a line with the bottom edge of the image
def find_intercept(height,top_margin,left_margin,x1,y1,x2,y2):
    '''
    find_intercept(height,top_margin,left_margin,x1,y1,x2,y2) function is used to determine intercepts of a given line with the bottom of the image

    input:  height - integer, height of the image on which the line lies
        top_margin,left margin - integers, the top and left margins of the region of interest in pixels
        x1,y1,x2,y2 - coordinates of two points which lie on the line for which intercepts need to be found
        
    output: intercept - float, the y coordinate where the line intersects the bottom line of the image.(origin beginning in the top left corner of the image)  
    '''
    Y1 = -(y1 + top_margin)
    X1 = x1 + left_margin
    Y2 = -(y2 + top_margin)
    X2 = x2+ left_margin
    Y = - height
    slope = float(Y2-Y1)/float(X2-X1)
    intercept = 0
    if slope != 0:
        intercept = (float(Y-Y1)/slope) + float(X1)
    return intercept

#function to determine the left and right lanes from a set of lines
def decide_lanes(left_or_right,lines,height,top_margin,left_margin,right_margin):
    '''
    decide_lanes(left_or_right,lines,height,top_margin,left_margin,right_margin) function is used to detect the left and right lanes out of all the lines detected. it uses a weighted score of distance between y intercept and mid point along with the steepness of the line to determine the lanes.

    input:  left_or_right - string, the choice of line which is to be detected i.e. "left"/"right"
        lines - list, a list of all input lines
        height - integer, heightof the image in pixels
        top_margin, left_margin, right_margin - integers, the top, left and right margins of the region of interest in pixels

    output:
        lane - list, has the structure [x1,y1,x2,y2,theta,intercept_with the bottom of the image]
    '''
    
    dist_weight = 2  	
    angle_weight = 10
    lane_score = -100000000
    lane =[]
    
    if left_or_right in ['left','Left','LEFT']:
        for x1,y1,x2,y2,rho,theta in lines:
            intercept = find_intercept(height,top_margin,left_margin,x1,y1,x2,y2) 
            if theta>0 and theta<np.pi/2: #the angle of left lane is assumed to be between 0and 90 degrees
                score = (-1*angle_weight*theta)+ (-1*dist_weight * (right_margin - intercept)) #lower the distance from center of the two lanes, higher the score, steeper the line higher the score
                if score > lane_score: #the line with highest score is the left lane
                    lane_score = score
                    lane = [x1+left_margin,y1+top_margin,x2+left_margin,y2+top_margin,theta,intercept]
        

    if left_or_right in ['right','Right','RIGHT']:
        for x1,y1,x2,y2,rho,theta in lines:
            intercept = find_intercept(height,top_margin,left_margin,x1,y1,x2,y2) 
            if theta < np.pi and theta > np.pi/2 : #the angle of right lane is assumed to be between 180 and 90 degrees
                score = (angle_weight*theta)+ (-1*dist_weight * (intercept - left_margin)) #lower the distance from center of two lanes, higher the score, steeper the line higher the score
                if score > lane_score: #the line with highest score is the right lane
                    lane_score = score
                    lane = [x1+left_margin,y1+top_margin,x2+left_margin,y2+top_margin,theta,intercept]

    return lane

#user defined function to calculate the dot product of two matrices
def dot_prod(a, b):#had to use a user defined function as numpy.dot was unreliable!
    '''
    dot_prod(a, b) function is used to determint the dot product between the matrices a and b. However the number of columns in a has to be equal to the number of rows in b, else an exception would be raised

    input:  a,b - numpy.ndarray, the two matrices whose dot product is to be determined
        
    output: dot - numpy.ndarray, the dot product of a and b

    '''
    if(a.shape[1]!=b.shape[0]):
        raise Exception("incompatible shapes",a.shape," ",b.shape)
        return 0
    else:
        
        dot= np.ones((a.shape[0],b.shape[1]))
        for i in range(0,a.shape[0]):
            for k in range(0,b.shape[1]):
                temp=0
                for j in range(0,a.shape[1]):
                    temp += a[i][j]*b[j][k]
                dot[i][k] = temp 
        return dot


#function to determine the state and covariance estimate of the system using Kalman's algorithm
def Kalman_filter_estimate(A,B,u,Ex,Q,P):
    '''
    Kalman_filter_estimate(A,B,u,Ex,Q,P) function is used to determint the state and the state covariance matrix of the described system using Kalman's algorithm

    input:  A - numpy.ndarray, State transition matrix of the system
        B - numpy.ndarray, Control correction matrix of the system
        u - numpy.ndarray, Control input to the systemz
        Ex - numpy.ndarray, State prediction error
        Q - numpy.ndarray, state of the system at the preceding time sample
        P - numpy.ndarray, state covariance matrix at the preceding time sample

    output: Q_est - numpy.ndarray, Estimated state at current time sample
        P_est - numpy.ndarray, Estimated state covariance matrix at current time sample 
    '''
    #Prediction part of Kalman Filter

    #Step 1: Est_of_x_at_t = (A * x_at_t-1) + (B * u_at_t-1)

    Q_est = dot_prod(A,Q) +  dot_prod(B,u) #prediction of state


    #Step 2: Est_of_state_cov_at_t = (A * state_cov_at_t-1 * A_transpose) + Ex

    P_est = dot_prod(dot_prod(A,P),A.transpose())+Ex #prediction of state covariance
    
    return (Q_est,P_est)

#function to determine the updated state and state covariance matrices of the system using Kalman's algorithm
def Kalman_Filter_Update(C,Ex,Ez,Q_est, P_est, z ):
    '''
    Kalman_Filter_Update(C,Ex,Ez,Q_est, P_est, z ) function is used to calculate the updated state and state covariance matix through feedback z using Kalman's algorithm

    input:  C - numpy.ndarray, Measurement conversion matrix of the system
        Ex - numpy.ndarray, State prediction error
        Ez - numpy.ndarray, Measurement error
        Q_est - numpy.ndarray, Estimated state for the current time sample
        P_est - numpy.ndarray, Estimated state covariance matrix for the current time sample
        z - numpy.ndarray, Measurement matrix

    output: Q - numpy.ndarray, Updated state of the system for the current time sample
        P - numpy.ndarray, Updated state covariance matrix for the crrent time step
    '''
    if Q_est.size:

        #Update part of Kalman filter

        #Step 3: Kalman_gain_at_t = Est_of_state_cov_at_t * C_transpose *inv( C * Est_of_state_cov_at_t * C_transpose   +  Ez  )

        K = dot_prod( dot_prod(P_est,C.transpose()), np.linalg.inv(dot_prod(C,dot_prod(P_est,C.transpose()))+Ez)  )# Kalman gain (The weightage that needs to be given for the discrepency in measurement and the prediction to update the state and its covariance )

        #Step 4: State_at_t = Est_of_state_at_t + Kalman_gain_at_t (measured_variable_at_t - (C * Est_of_state_at_t) )

        Q = Q_est + dot_prod(K , (z - dot_prod(C,Q_est)  )  ) # correcting state estimate using measured variables to obtain actual state

        #Step 5: State_cov_at_t = (I - Kalman_gain_at_t * C) Est_of_state_cov_at_t

        P = dot_prod( (np.identity(4) - dot_prod(K,C)  ), P_est)# correcting state covariance estimate using measured variables to obtain actual state covariance 

    else:
        Q = np.array([[z[0][0]],[z[1][0]],[0],[0]])#state variable [[c],[theta],[c'],[theta']]
        P = Ex #Estimate of initial state covriance matrix
    return (Q,P)


#function to eliminate background using estimated states to reduce process time
def elimnate_predicted_background(lane, img, img_width,img_height,top_margin, left_margin,  height, Q_est, prediction_error_tolerance ):
    '''
    elimnate_predicted_background(lane,img,img_width,img_height,top_margin,left_margin,height,Q_est,prediction_error_tolerance) function is used to eliminate background using state estimate to improve performance. only a thin strip around the estimate of width equal to error tolerance is left

    input:  lane - string, Choice of the lane which is to be processed i.e. "left"/"right" 
        img - numpy.ndarray, the image in which the background is to be removed
        img_width - int, Width of the image, which is to be processed
        img_height - int, Height of the image, which is to be processed
        top_margin, left_margin - ints, top and left margins used to obtain the current image from the original image based on which intercepts are calculated
        height - int, height of the original image based on which intercepts are calculated 
        Q_est - numpy.ndarray, Estimated state for the current time sample
        prediction_error_tolerance - int, the tolerance in pixels afforded to the estimated state

    output: img - numpy.ndarray, the image where background is blacked out to improve performance

    '''
    x1 = Q_est[0][0]
    y1 = height
    x2 = 0
    y2 = height+(Q_est[0][0]/np.tan(Q_est[1][0]) )
    
    if (lane in ["left", "Left","LEFT"]):
        for x in range(0,img_width):
            for y in range(0,img_height):
                if y<(y1  -  top_margin - (((x+left_margin) - (x1 - prediction_error_tolerance))*(y1-y2) /(x2-x1)   )  ) or y>(y1  -  top_margin - (((x+left_margin) - (x1 + prediction_error_tolerance))*(y1-y2) /(x2-x1)   )  )   :
                    img[y][x] = 0
    if (lane in ["right", "Right","RIGHT"]):
        for x in range(0,img_width):
            for y in range(0,img_height):
                if y>(y1  -  top_margin - (((x+left_margin) - (x1 - prediction_error_tolerance))*(y1-y2) /(x2-x1)   )  ) or y<(y1  -  top_margin - (((x+left_margin) - (x1 + prediction_error_tolerance))*(y1-y2) /(x2-x1)   )  )   :
                    img[y][x] = 0
    

    return img

#function to detect lines by iteratively decreasing Hough threshold untill atleast one line is detected
def detect_best_lines(img,Hough_threshold_upper_limit,Hough_threshold_lower_limit,lower_angle_lim, upper_angle_lim):
    '''
    detect_best_lines(img,Hough_threshold_upper_limit,Hough_threshold_lower_limit,lower_angle_lim, upper_angle_lim) function is used to detect lines by iteratively decreasing Hough threshold untill atleast one line is detected

    input: 	img - numpy.ndarray, the image on which lines are to be detected
        Hough_threshold_upper_limit - int, The threshold with which the iterative reduction begins
        Hough_threshold_lower_limit - int, The lowest threshold to which iterative reduction is done untill a line is found
        lower_angle_lim, upper_angle_lim - floats, Angles between which the detected lines should lie for them to be considered as potential lane markings

    output: lines - list, the list of all lines that are detected at highest possible threshold
    '''
    Hough_threshold = Hough_threshold_upper_limit
    lines = line_detect(img, Hough_threshold,lower_angle_lim, upper_angle_lim )
    while (isinstance(lines,list) and Hough_threshold>Hough_threshold_lower_limit):
        lines = line_detect(img, Hough_threshold,lower_angle_lim, upper_angle_lim )
        Hough_threshold -= 5
        pass
    return lines




def ImShow(img,title="", cmap=None):
    plt.figure(figsize=(15,15))
    plt.title(title)
    plt.imshow(img,cmap=cmap)
    plt.show()
cap = cv2.VideoCapture("ver1.mp4")
while(cap.isOpened()):
    ret,image1 = cap.read()
    img= image1[10:200,10:300]
    top_margin = 0#pixels
    bottom_margin = 100-10#pixels
    left_lane_left_margin = 0#pixels
    left_lane_right_margin = 150-10#pixels
    right_lane_left_margin = 150-10#pixels
    right_lane_right_margin = 300-10#pixels
    intercepts = []
    predictions = []
    predictions.append(("Left_lane_intercept_estimate","Left_lane_intercept_actual", "Left_lane_intercept_measured", "Left_lane_angle_estimate","Left_lane_angle_actual", "Left_lane_angle_measured","Right_lane_intercept_estimate","Right_lane_intercept_actual", "Right_lane_intercept_measured", "Right_lane_angle_estimate","Right_lane_angle_actual", "Right_lane_angle_measured"))
    prediction_error_tolerance = 10 #pixels
    left_lane = []
    right_lane = []
    left_Q_est = np.array([])
    left_P_est = np.array([])
    right_Q_est = np.array([])
    right_P_est = np.array([])
    dt = 0.1
    u = np.array([[0.01],[0.01]])
    acc_noise = 0.1
    c_meas_noise = 0.1
    theta_meas_noise = 0.1
    Ez = np.array(   [[c_meas_noise,0],[0,theta_meas_noise]] )#measurement prediction error
    Ex = np.array( [[(dt**4)/4,0,(dt**3)/2,0],[0,(dt**4)/4,0,(dt**3)/2],[(dt**3)/2,0,(dt**2),0],[0,(dt**3)/2,0,(dt**2)]])* (acc_noise**2)#State prediction error
    #state and measurement equations
    A = np.array([[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]] )
    B = np.array([[(dt**2)/2,0],[0,(dt**2)/2],[dt,0],[0,dt]] )
    C = np.array([[1,0,0,0],[0,1,0,0]] )

    
    if left_lane:

            (left_Q_est,left_P_est) = Kalman_filter_estimate(A,B,u,Ex,left_Q,left_P)



        
    if right_lane:

            (right_Q_est,right_P_est) = Kalman_filter_estimate(A,B,u,Ex,right_Q,right_P)
    trial = 1
    left_lane = []
    right_lane = []
        
    while ((not left_lane) or (not right_lane)):

        #image processing
        imgg=process_img(img)
        left_lane_img = imgg[top_margin:bottom_margin,left_lane_left_margin:left_lane_right_margin]
        right_lane_img = imgg[top_margin:bottom_margin,right_lane_left_margin:right_lane_right_margin]
        left_img_pr = left_lane_img
        right_img_pr = right_lane_img

        #Elimination of Background based on Kalman Filtering
    
        if left_Q_est.size and trial ==1:
            left_img_pr = elimnate_predicted_background("left", left_img_pr, left_lane_img.shape[1],left_lane_img.shape[0], top_margin, left_lane_left_margin,  height, left_Q_est, prediction_error_tolerance )

    
        
        if right_Q_est.size and trial ==1:

            right_img_pr = elimnate_predicted_background("right", right_img_pr, right_lane_img.shape[1], right_lane_img.shape[0], top_margin, right_lane_left_margin,  height, right_Q_est, prediction_error_tolerance )

        

        #determination of best estimates of left and right line through a combination of best detectable lines and a weighted score on their anglesand intercepts with the bottomof the picture
        if(not left_lane):
            left_lines = detect_best_lines(left_img_pr,170,90,75,105)	
            left_lane = decide_lanes("left",left_lines,img.shape[0],top_margin, left_lane_left_margin, left_lane_right_margin )

        if(not right_lane):
            right_lines = detect_best_lines(right_img_pr,170,90,75,105)
            right_lane = decide_lanes("right",right_lines,img.shape[0], top_margin, right_lane_left_margin, right_lane_right_margin )

        #best detected lines for debugging
        #left_lines_detected_img = cv2.cvtColor(left_img_pr,cv2.COLOR_GRAY2BGR)
        #right_lines_detected_img = cv2.cvtColor(right_img_pr,cv2.COLOR_GRAY2BGR)
    
        #if isinstance(left_lines,list):# to make sure the list is not empty
        #	draw_line(left_lines_detected_img,left_lines,(0,255,0))#green
        #if isinstance(right_lines,list):# to make sure the list is not empty
        #	draw_line(right_lines_detected_img,right_lines,(0,255,0))#green
    
        #show all detected lines
            
            #cv2.imshow('left',left_lines_detected_img)
            #cv2.imshow('right',right_lines_detected_img)
        #end of debugging code

        #measured intercept and angle values are used to update Kalman filter
        if left_lane:

            (left_Q,left_P) = Kalman_Filter_Update(C,Ex,Ez,left_Q_est, left_P_est, np.array([[left_lane[-1]],[left_lane[4]]]) )

        if right_lane:

            (right_Q,right_P) = Kalman_Filter_Update(C,Ex,Ez,right_Q_est, right_P_est, np.array([[right_lane[-1]],[right_lane[4]]]) )
        if(trial >=3):
            if not left_lane:
                
                x1 = left_Q_est[0][0]
                y1 = height
                x2 = 0
                y2 = height+(left_Q_est[0][0]/np.tan(left_Q_est[1][0]) )
                theta = left_Q_est[1][0]
                intercept = left_Q_est[0][0]
                left_lane = [x1,y1,x2,y2,theta,intercept]
            if not right_lane:
                x1 = right_Q_est[0][0]
                y1 = height
                x2 = 0
                y2 = height+(right_Q_est[0][0]/np.tan(right_Q_est[1][0]) )
                theta = right_Q_est[1][0]
                intercept = right_Q_est[0][0]
                right_lane = [x1,y1,x2,y2,theta,intercept]

        trial +=1 
    

        pass
    # Draw sample lane markers
        (height, width) = img.shape[:2]
        color = (0,255,255) # yellow
    if len(left_lane) == 0 :    
        left_x = 'None'		
    else:
        left_x = left_lane[-1]
        cv2.line(img, (int(left_lane[0]),int(left_lane[1])), (int(left_lane[2]), int(left_lane[3])), color,5)
    if len(right_lane)== 0:
        right_x = 'None'
    else:
        right_x = right_lane[-1]
        cv2.line(img, (int(right_lane[0]),int(right_lane[1])), (int(right_lane[2]), int(right_lane[3])), color,5)

    
    # Sample intercepts
        #intercepts.append((os.path.basename(fname), left_x, right_x))
    if (left_Q_est.size and right_Q_est.size and left_Q.size and right_Q.size ):		
        predictions.append((left_Q_est[0][0], left_Q[0][0], left_lane[-1], left_Q_est[1][0], left_Q[1][0], left_lane[4], right_Q_est[0][0], right_Q[0][0], right_lane[-1], right_Q_est[1][0], right_Q[1][0], right_lane[4]))

        # Show image
        cv2.imshow('Lane Markers', imgg)
        key = cv2.waitKey(100)

    # lines = cv2.HoughLinesP(line_segmented, rho=1, theta=np.pi/180, threshold=30, minLineLength=10, maxLineGap=30)
    # left_lines    = [] # (slope, intercept)
    # right_lines   = [] # (slope, intercept)
    # try:
    # 	for line in lines:
    # 		for x1, y1, x2, y2 in line:
    # 			if x1 == x2:
    # 				continue
    # 			slope = (y2-y1)/(x2-x1)
    # 			intercept = y1 - slope*x1
    # 			length = np.sqrt((y2-y1)**2+(x2-x1)**2)

    # 			if slope < -0.4 and intercept < 500: # y is reversed in image
    # 				left_lines.append([slope, intercept])
    # 			elif 5 > slope > 0.4 and intercept < 500:
    # 				right_lines.append([slope, intercept])
    # 			else:
    # 				continue
    # except:
    # 	pass


    # left_lines = np.asarray(left_lines)
    # right_lines = np.asarray(right_lines) 

    # if len(left_lines) > 0:
    #     #left_lines = [np.mean(left_lines[:,0]), np.mean(left_lines[:,1])]
    #     [a_l, b_l] = left_lines[0]
    #     cv2.line(image[int(h*0.5):h, :], (int((h-b_l)/a_l), h), ( int((h*0.1-b_l)/a_l), int(h*0.1)), (0,255,0), 5 )
    #     #print("Phương trình đường thẳng line trái y = {0:.2f} * x + {0:.2f}".format(left_lines[0][0],left_lines[0][1]))
    # if len(right_lines) > 0:
    #     #right_lines = [np.mean(right_lines[:,0]), np.mean(right_lines[:,1])]
    #     [a_r, b_r] = right_lines[0]
    #     cv2.line(image[int(h*0.5):h, :], (int((h-b_r)/a_r), h), ( int((h*0.1-b_r)/a_r), int(h*0.1)), (0,255,0), 5 )
    #     #print("Phương trình đường thẳng line phải y =  {0:.2f} * x + {0:.2f}".format(right_lines[0][0],right_lines[0][1]))


    cv2.imshow("Visualize Original Image",img)
    if cv2.waitKey(25) & 0xff == ord('q'):
        cv2.destroyAllWindows()
        break    
cap.release()
cv2.destroyAllWindows()
