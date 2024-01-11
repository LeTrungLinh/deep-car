import torch
import cv2
import time
import numpy as np
from PIL import Image
from torchvision import transforms
from models.ex4 import BiSeNet
from models_utils.helpers import load_checkpoint

mapping = {
	(31,120,180):0, #road
	(227,26,28) :1, #people
	(106,61,154):2, #car
	(0, 0, 0)   :3, #background
	}
rev_mapping = {mapping[k]: k for k in mapping}

model = BiSeNet(4)
load_checkpoint(torch.load('checkpoints/best_model_ute.pth'), model)
model = model.cuda().eval()

def preprocess(image):
    image = Image.fromarray(image)
    data_transforms = transforms.Compose([
        transforms.Resize((320,320)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    image = data_transforms(image)
    image = image.float()
    image = image.cuda()
    image = image.unsqueeze(0)
    return image

cap = cv2.VideoCapture('video/vid2.avi')
while cap.isOpened():
    t1 = time.time()
    _, frame = cap.read()
    image = frame.copy()
    image_data = preprocess(image)
    shit = model(image_data)
    prediction = torch.argmax(shit,1)[0]
    pred_image = torch.zeros(3, prediction.size(0), prediction.size(1), dtype=torch.uint8)
    for k in rev_mapping:
        pred_image[:, prediction==k] = torch.tensor(rev_mapping[k]).byte().view(3, 1)
    road_mask = pred_image.permute(1, 2, 0).numpy()
    road_mask = cv2.cvtColor(road_mask, cv2.COLOR_RGB2BGR)
    cv2.imshow('hey', road_mask)
    print("fps",1/(time.time()-t1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break