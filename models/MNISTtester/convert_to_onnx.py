import torch
import sys

from .. import CNN

PACKAGE_PARENT = '..'

def main():
    print("Loading model")
    pytorch_model = CNN.LeNet5()
    print("Loading weight data")
    pytorch_model.load_state_dict(torch.load('models/LeNet5'))
    pytorch_model.eval()
    dummy_input = torch.zeros(size=(1,1, 28, 28))
    print("Exporting model")
    torch.onnx.export(pytorch_model, dummy_input, 'models/MNISTtester/onnx_model.onnx', verbose=True)