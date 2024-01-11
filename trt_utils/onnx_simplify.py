import onnx
from onnxsim import simplify

# load your predefined ONNX model
model = onnx.load('bisenet.onnx')

# convert model
model_simp, check = simplify(model)

assert check, "Simplified ONNX model could not be validated"

# use model_simp as a standard ONNX model object