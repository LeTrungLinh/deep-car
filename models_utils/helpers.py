import os 
import torch
import numpy as np
from torch import nn
from PIL import Image
from glob import glob
from torchvision import models
import torch.nn.functional as F
# Build data loader
from tqdm import tqdm
from torchvision import transforms
from collections import OrderedDict
from torch.utils.data import DataLoader, Dataset
# from matplotlib import pyplot as plt

def reverse_one_hot(image):
    image = image.permute(1,2,0)
    x = torch.argmax(image, dim=-1)
    return x

def compute_accuracy(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    # print(label)
    total = len(label)
    # print(total)
    count = 0.0
    for i in range(total):
        if pred[i] == label[i]:
            count == count + 1.0
    return float(count)/float(total)

def fast_hist(a,b,n):
    k = (a>=0)&(a<n)
    return np.bincount(n*a[k].astype(int) + b[k], minlength=n**2).reshape(n,n)

def per_class_iu(hist):
    epsilon = 1e-5
    return (np.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

def save_checkpoint(state, filename):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def colour_code_segmentation(image, label_values):
    w = image.shape[0]
    h = image.shape[1]
    x = np.zeros([w,h,3], dtype=np.uint8)
    colour_codes = label_values
    
    for i in range(0, w):
        for j in range(0, h):
            x[i, j, :] = colour_codes[int(image[i, j])]
    return x

