from xmlrpc.client import TRANSPORT_ERROR
import torch
from models.ex4 import BiSeNet
from models_utils.helpers import load_checkpoint
from addict import Dict

# model = BiSeNet(5)
model=BiSeNet(4).cuda()
load_checkpoint(torch.load('checkpoints/best_model_ute110_bisenet18_360_360_2_7.pth'), model)
dummy_input = torch.randn(1, 3, 360, 360).cuda()
input_names = [ "image" ]
output_names = [ "probabilities" ]
torch.onnx.export(model, 
                  dummy_input,
                  "checkpoints_trt/bisenet18_2_7.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names,
                  opset_version=12
                  )