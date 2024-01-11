import torch
import numpy as np
import cv2
# from adafruit_servokit import ServoKit
from PIL import Image
from torchvision import transforms
from torch.serialization import load
from models.ex4 import BiSeNet
# from models.bisenet_v6 import BiSeNet

import time
from models_utils.helpers import load_checkpoint
# from drive_utils.drivingv2 import *
import drive_utils.drivingv3 as st_utils
from drive_utils.calib_utils import distort_calib2
# cuDnn configurations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
count=0
# kit = ServoKit(channels=16)
model = BiSeNet(5)
# load_checkpoint(torch.load('checkpoints/best_model_ute110_bisenetv4_360.pth'), model)
load_checkpoint(torch.load('checkpoints/best_model_ute110_bisenet_12_3_360_2.pth'), model)

data_transforms = transforms.Compose([
	transforms.Resize((360,360)),
	# transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
	transforms.ToTensor(),
	transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

device = torch.device("cuda:0")
model = model.to(device)
model.eval()

def preprocess(image):
	image = Image.fromarray(image)
	image = data_transforms(image)
	image = image.float()
	image = image.to(device)
	image = image.unsqueeze(0)
	return image

# cap = cv2.VideoCapture(1, cv2.CAP_V4L2)
cap = cv2.VideoCapture('video/project2.avi')
# cap = cv2.VideoCapture(" v4l2src device=/dev/video1 ! image/jpeg, format=MJPG, width=1280, height=720 ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1", cv2.CAP_GSTREAMER)
# cap.set(cv2.CAP_PROP_FPS, 30)
# cap.set(3, 640)
# cap.set(4, 480)

##reverse mapping
mapping = {
	(31,   120,   180):0, #road
	(227,  26,    288) :1, #mid
	(106, 61, 154):2, #car
	(255,255,153):3, #person
	(0, 0, 0)   :4, #background
	}
rev_mapping = {mapping[k]: k for k in mapping}
##
while True:
	t=time.time()
	ret, frame = cap.read()
	
	# frame = cv2.flip(frame, 0)
	# frame = cv2.flip(frame,1)
	image = frame.copy()
	# if count%2==0:
	# image = distort_calib2(image)
	image_data = preprocess(image)
	shit = model(image_data)
	prediction = torch.argmax(shit,1)[0]
	
	pred_image = torch.zeros(3, prediction.size(0), prediction.size(1), dtype=torch.uint8)
	for k in rev_mapping:
		pred_image[:, prediction==k] = torch.tensor(rev_mapping[k]).byte().view(3, 1)
	road_mask = pred_image.permute(1, 2, 0).numpy()
	# road_mask = cv2.cvtColor(road_mask, cv2.COLOR_RGB2BGR)

	#   DISTANCE MATRIX: MA TRẬN KHOẢNG CÁCH - TÍNH TÂM ĐƯỜNG -> GÓC LÁI
	#tính ma trận tâm đường
	# stack, mask = dmatrix(image,road_mask)
	# cv2.imshow('results', stack)
	# land_follower = st_utils.HandCodedLaneFollower()
	# combo_image = land_follower.follow_lane(road_mask,image)
	#steering angle
	# angle = steering(road_mask) #mask,kp,ki,kd
	# kit.servo[0].angle=angle
	# print('goc lai:', angle)
	cv2.imshow("frame", image)
	cv2.imshow("mask", road_mask)
	print("fps",1/(time.time()-t))

	
	if cv2.waitKey(1) & 0xFF == ord('q'):
        	break
	
cap.release()



