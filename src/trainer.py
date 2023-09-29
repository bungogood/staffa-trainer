import torch
from torch import nn
from model import Model
import onnx2torch
import numpy as np

# load onnx model
def load_model(path, as_onnx=True):
    if as_onnx:
        model = onnx2torch.convert(path)
    else:
        model = Model()
        model.load_state_dict(torch.load(path))
    return model

# load onnx model
def save_model(model, path, as_onnx=True):
    if as_onnx:
        dummy_input = torch.randn(1, 202)
        torch.onnx.export(model, dummy_input, path, export_params=True, verbose=False)
    else:
        torch.save(model.state_dict(), path)
    return model

def read_csv(path):
    data = []
    labels = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.strip().split(';')
            line = list(map(float, line))
            labels.append(line[:6])
            data.append(line[6:])
    return np.array(data), np.array(labels)

def train(model, data, labels):
    return model


def main(model, output_path, data_path):
    # Read CSV file
    data, labels = read_csv(data_path)
    print(data.shape, labels.shape)

    # Train model
    model = train(model, data, labels)

    save_model(model, output_path)

if __name__ == '__main__':
    onnx_path = 'model/model.onnx'
    output_path = 'model/model-01.onnx'
    data_path = 'data/rollouts-07.csv'
    if False:
        model = load_model(onnx_path)
    else:
        model = Model()
    main(model, output_path, data_path)
