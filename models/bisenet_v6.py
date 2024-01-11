# BISENET RIP OFF LAST CONV IN RESNET LAYER4
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

from models.xception import xception39

# Define the base Convolution block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        # Head of block is a convulution layer
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        # After conv layer is the batch noarmalization layer 
        self.batch_norm = nn.BatchNorm2d(out_channels)
        
        # Tail of this block is the ReLU function 
        self.relu = nn.ReLU()   
        
    def forward(self, x):
        # Main forward of this block 
        x = self.conv1(x)
        x = self.batch_norm(x)
        return self.relu(x)

# Define the Spatial Path with 3 layers of ConvBlock 

class SpatialPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)



# Attention Refinement Module 

class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        
    def forward(self, x_input):
        # Apply Global Average Pooling
        x = self.avg_pool(x_input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        x = self.bn(x)
        x = self.sigmoid(x)
        
        # Channel of x_input and x must be same 
        return torch.mul(x_input, x)

# Define Feature Fusion Module 
class FeatureFusionModule(nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.conv_block = ConvBlock(in_channels=in_channels, out_channels=num_classes, stride=1)
        self.conv1 = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            
    def forward(self, x_input_1, x_input_2):
        x = torch.cat((x_input_1, x_input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.conv_block(x)
        
        # Apply above branch in feature 
        x = self.avg_pool(feature)
        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        
        # Multipy feature and x 
        x = torch.mul(feature, x)
        
        # Combine feature and x
        return torch.add(feature, x)

# Build context path 
class ContextPath(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # RESNET18
        # self.features = models.resnet18(pretrained=pretrained)
        # self.conv1 = self.features.conv1
        # self.bn1 = self.features.bn1
        # self.relu = self.features.relu
        # self.max_pool = self.features.maxpool
        # self.layer1 = self.features.layer1
        # self.layer2 = self.features.layer2
        # self.layer3 = self.features.layer3
        # self.layer4 = self.features.layer4[0]
        # self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # EXCEPTION39
        self.xception39 = xception39(pretrained_model=None, norm_layer=nn.BatchNorm2d)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        
    def forward(self, x_input):
        # Get feature from lightweight backbone network

        feature = self.xception39(x_input)
        feature3 = feature[2]
        feature4 = feature[3]
        tail = self.avg_pool(feature[3])
        # print(tail.shape)
        # print('context block', context_block[0].shape)
        # print('context block1', context_block[1].shape)
        # print('context block2', context_block[2].shape)
        # print('context block3', context_block[3].shape)
        return feature3, feature4, tail

# Define BiSeNet 

class BiSeNet(nn.Module):
    def __init__(self, num_classes, training=True):
        super().__init__()
        self.training = training
        self.spatial_path = SpatialPath()
        self.context_path = ContextPath()
        self.arm1 = AttentionRefinementModule(in_channels=128, out_channels=128)
        self.arm2 = AttentionRefinementModule(in_channels=256, out_channels=256)
        
        # Supervision for calculate loss 
        self.supervision1 = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=1)
        self.supervision2 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
        
        # Feature fusion module 
        self.ffm = FeatureFusionModule(num_classes=num_classes, in_channels=640)
        
        # Final convolution 
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)
        
    def forward(self, x_input):
        # print('\nBISENET WITH XCEPTION39 BACKBONE')
        # Spatial path output
        sp_out = self.spatial_path(x_input)
        
        # Context path output
        feature3, feature4, tail = self.context_path(x_input)
        
        # apply attention refinement module 
        feature3, feature4 = self.arm1(feature3), self.arm2(feature4)
        
        # Combine output of lightweight model with tail 
        feature4 = torch.mul(feature4, tail)
        
        # Up sampling 
        size2d_out = sp_out.size()[-2:]
        feature3 = F.interpolate(feature3, size=size2d_out, mode='bilinear')
        feature4 = F.interpolate(feature4, size=size2d_out, mode='bilinear')
        context_out = torch.cat((feature3, feature4), dim=1)
        
        # Apply Feature Fusion Module 
        combine_feature = self.ffm(sp_out, context_out)
        
        # Up sampling 
        bisenet_out = F.interpolate(combine_feature, scale_factor=8, mode='bilinear')
        bisenet_out = self.conv(bisenet_out)
        
        # When training model 
        if self.training is True:
            feature3_sup = self.supervision1(feature3)
            feature4_sup = self.supervision2(feature4)
            feature3_sup = F.interpolate(feature3_sup, size=x_input.size()[-2:], mode='bilinear')
            feature4_sup = F.interpolate(feature4_sup, size=x_input.size()[-2:], mode='bilinear')        
            return bisenet_out, feature3_sup, feature4_sup
        return bisenet_out

# TEST MODEL
# bisenet = BiSeNet(num_classes=2, training=True)
# bisenet = bisenet.cuda()
# output, output_sup1, output_sup2 = bisenet(torch.rand((8, 3, 360, 360)).cuda())
# print(output.shape)

# from torchsummary import summary
# summary(bisenet, (3, 360, 640))


