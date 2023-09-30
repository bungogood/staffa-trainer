from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import pandas as pd
from torch import nn
import torch

class Model(nn.Module, ABC):
    def input_size(self) -> int:
        first, *_ = self.parameters()
        return first.size()[1]
    
    def output_size(self) -> int:
        *_, last = self.parameters()
        return last.size()[0]

    def save_onnx(self, path: str) -> None:
        dummy_input = torch.randn(1, self.input_size(), requires_grad=True)
        torch.onnx.export(self, dummy_input, path)

class ModelDataset(Dataset):
    def __init__(self, model: Model, csv_files: list | str):
        if isinstance(csv_files, str):
            csv_files = [csv_files]
        data = pd.concat([pd.read_csv(f) for f in csv_files ], ignore_index=True)
        input_size = model.input_size()
        output_size = model.output_size()
        if data.shape[1] != input_size + output_size:
            raise ValueError(f"Expected {input_size + output_size} columns, got {data.shape[1]}")
        self.inputs = torch.Tensor(data.iloc[:, output_size:].values)
        self.labels = torch.Tensor(data.iloc[:, :output_size].values)
    
    def to(self, device):
        self.inputs = self.inputs.to(device)
        self.labels = self.labels.to(device)
        return self

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]
