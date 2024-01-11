import onnx
import cv2
import torch
import numpy as np
import time
from PIL import Image
from torchvision import transforms
# onnx_model = onnx.load('bisenet.onnx')
# onnx.checker.check_model(onnx_model)
# print('done')


def preprocess(image):
    input_img = image.astype(np.float32)
    img_height = 320
    img_width = 320
    mean = np.array([0.485, 0.456, 0.406]) * 255.0
    scale = 1 / 255.0
    std = [0.229, 0.224, 0.225]
    input_blob = cv2.dnn.blobFromImage(
        image=input_img,
        scalefactor=scale,
        size=(img_width, img_height),  # img target size
        mean=mean,
        swapRB=False,  # BGR -> RGB
        crop=False  # center crop
    )
    input_blob[0] /= np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
    return input_blob

# cap = cv2.VideoCapture('video/vid2.avi')
cap = cv2.VideoCapture(" v4l2src device=/dev/video1 ! image/jpeg, format=MJPG, width=1280, height=720 ! nvv4l2decoder mjpeg=1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1", cv2.CAP_GSTREAMER)

opencv_net = cv2.dnn.readNetFromONNX('bisenetv44.onnx')
opencv_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
opencv_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
# print("OpenCV model was successfully read. Layer IDs: \n", opencv_net.getLayerNames())

while cap.isOpened():
    # read the image
    # image = cv2.imread('data/data_801.png')
    t1 = time.time()
    _, frame = cap.read()
    image = frame.copy()
    inputb = preprocess(image)
    opencv_net.setInput(inputb)
    out = opencv_net.forward()
    print("* shape: ", out.shape)

    # get IDs of predicted classes
    prediction = np.argmax(out[0], axis=0)
    mapping = {
        (31,120,180):0, #road
        (227,26,28) :1, #people
        (106,61,154):2, #car
        (0, 0, 0)   :3, #background
        }
    rev_mapping = {mapping[k]: k for k in mapping}

    pred_image = np.zeros((3, prediction.shape[0], prediction.shape[1]))
    for k in rev_mapping:
        pred_image[:, prediction==k] = torch.tensor(rev_mapping[k]).byte().view(3, 1)
    road_mask = np.transpose(pred_image, (1, 2, 0))
    print(road_mask.shape)
    cv2.imshow('hey', road_mask.astype('uint8'))
    cv2.imshow('frame', frame)
    print("fps",1/(time.time()-t1))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
	
cap.release()