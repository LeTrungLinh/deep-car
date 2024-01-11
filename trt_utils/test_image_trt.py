from numpy import full
import os
import cv2
import config as cf
from numpy import interp
#---------My libs---------#
from PIL import Image
from bisenet_trt import TRTSegmentor
from trt_utils.segcolors import lanecolor,midcolor,colors
#---------Camera params---------#

    
if __name__ == '__main__':
    # INIT VIDEO CAPTURE OPJECT
    bisenet_trt=TRTSegmentor('checkpoints_trt/bisenet5_15_3_2.onnx', 
        colors,
        lanecolor,
        midcolor, 
        device='GPU', 
        precision='FP16',
        calibrator=None, 
        dla_core=0)
    image_dir = 'test_images'
    images = os.listdir(image_dir)
    for idx in range(len(images)):
        img_path = os.path.join(image_dir, images[idx])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (360,360))
        duration = bisenet_trt.infer(img, benchmark=True)
        lanemask, midmask, obstacle, fullmask = bisenet_trt.draw(img)
        cv2.imshow("mask", cv2.cvtColor(fullmask, cv2.COLOR_BGR2RGB))
        cv2.imshow("frame", img)
        # cv2.imshow('obstacle', obsmask)
        # if cf.show:
        #     stacked_image = dmatrix(image,fullmask)
        #     cv2.imshow('results', stacked_image)
        cv2.waitKey(0)