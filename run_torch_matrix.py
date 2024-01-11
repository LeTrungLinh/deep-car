import sys
import numpy as np
import cv2
import torch
from numpy import interp
#---------My libs---------#
from drive_utils.mask2bbox import mask_to_bbox
from drive_utils.driving import steering
from models_utils.helpers import load_checkpoint
from models.ex4 import BiSeNet
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
#---------Servo libs---------#
#---------Run configs---------#
#---------BISENET LANE DETECTION---------#
def bisenet_pipeline():
        frame = cv2.imread('383.jpg')
        img = frame.copy()
        ##transform
        transform = A.Compose([
            A.Resize(height=360, width=360),
            A.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
            ToTensorV2(),
        ]) ##imagenet norm
        augmentations = transform(image=img)
        input = augmentations["image"].unsqueeze(0)
        output = model(input).to(device='cuda')
        output = torch.argmax(output,1)[0]

        mapping = {
            (31,120,180):0,
            (227,26,28):1,
            (106,61,154):2,
            (0,0,0):3,
            }
        rev_mapping = {mapping[k]: k for k in mapping}

        pred_image = torch.zeros(3, output.size(0), output.size(1), dtype=torch.uint8)
        print('pred_image',pred_image.size())
        for k in rev_mapping:
            pred_image[:, output==k] = torch.tensor(rev_mapping[k]).byte().view(3, 1)
        final_img = pred_image.permute(1, 2, 0).numpy()
        fullmask = Image.fromarray(final_img,'RGB')

        error, angle, stack_img = steering(img, final_img) #mask,kp,ki,kd
        sangle  = interp(angle,[-30,30],[30,-30])
        aglservo = interp(angle,[-30,30],[500,300]) # map angle to servo
        #---------READ GPS SIGNALS FROM TXT FILE---------#
        with open('sensor_log/eyaw.txt') as file:
            lines = file.readlines()
        print('eyaw from gps',lines)
        print('eyaw from vision',sangle)
        #---------SEND ANGLE TO SERVO---------#
        #---------SHOW RESULTS---------#
        cv2.imshow("frame", stack_img)
        # cv2.imshow('obstacle', obsmask)
        # if cf.show:
        #     stacked_image = dmatrix(image,fullmask)
        #     cv2.imshow('results', stacked_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        

if __name__ == '__main__':
    # INIT VIDEO CAPTURE OPJECT
    # THREADED RUNS
    # hough_thread = threading.Thread(name='hough_trans', target=hough_pipeline)
    # hough_thread.start()
    # segment_thread = threading.Thread(name='lane_detect', target=bisenet_pipeline())
    # segment_thread.start()
    model = BiSeNet(4)
    load_checkpoint(torch.load('checkpoints/best_model_ute110_bisenet18_360_360_2_7.pth'), model)
    model.eval()
    # NORMAL RUNS
    bisenet_pipeline()
