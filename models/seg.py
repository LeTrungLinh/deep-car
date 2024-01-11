import torch
from torch import nn
from torchvision import models
import torchvision.transforms as T
import numpy as np
import cv2
import time
from segcolors import colors

class SegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net= models.segmentation.fcn_resnet50(pretrained=True, aux_loss=False).cuda()
        self.ppmean=torch.Tensor([0.485, 0.456, 0.406])
        self.ppstd=torch.Tensor([0.229, 0.224, 0.225])
        self.preprocessor=T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        self.cmap=torch.from_numpy(colors[:,::-1].copy())

    def forward(self, x):
        """x is a pytorch tensor"""

        #x=(x-self.ppmean)/self.ppstd #uncomment if you want onnx to include pre-processing
        isize=x.shape[-2:]
        x=self.net.backbone(x)['out']
        x=self.net.classifier(x)
        #x=nn.functional.interpolate(x, isize, mode='bilinear') #uncomment if you want onnx to include interpolation
        return x

if __name__=='__main__':
    model=SegModel()
    x=torch.randn(1,3,320,320).cuda() #360p size
    output=model(x)
    print(output.shape)
    # model.export_onnx('./segmodel.onnx')