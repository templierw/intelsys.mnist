import torch
import sys

from CNN import LeNet5

device = torch.device('cpu')
print("Loading model")
pytorch_model = LeNet5()
print("Loading weight data")
pytorch_model.load_state_dict(torch.load('saved/LeNet5', map_location=device))
pytorch_model.eval()
dummy_input = torch.zeros(size=(1,1, 28, 28))
print("Exporting model")
torch.onnx.export(pytorch_model, dummy_input, '../MNISTtester/onnx/onnx_model.onnx', verbose=True)
