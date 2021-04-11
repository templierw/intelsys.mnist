import torch
import sys

from CNN import *
from MLP import *

model = LeNet5() # check MLP.py and CNN.py for more models
name = "LeNet5"

device = torch.device('cpu')
print("Loading model")
pytorch_model = model
print("Loading weight data")
pytorch_model.load_state_dict(torch.load(f'saved/{name}', map_location=device))
pytorch_model.eval()
dummy_input = torch.zeros(size=(1,1, 28, 28))
print("Exporting model")
torch.onnx.export(pytorch_model, dummy_input, f'../docs/onnx/{name}.onnx', verbose=True)
