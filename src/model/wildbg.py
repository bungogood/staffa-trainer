from torch.utils.data import Dataset
from torch import nn
import pandas as pd
import torch

class WildBgDataSet(Dataset):
    def __init__(self, csv_files: list | str):
        if isinstance(csv_files, str):
            csv_files = [csv_files]
        data = pd.concat([pd.read_csv(f) for f in csv_files ], ignore_index=True)
        self.inputs = torch.Tensor(data.iloc[:, 6:].values)
        self.labels = torch.Tensor(data.iloc[:, :6].values)
    
    def to(self, device):
        self.inputs = self.inputs.to(device)
        self.labels = self.labels.to(device)
        return self

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        # First 6 columns are outputs, last 202 columns are inputs
        return self.inputs[idx], self.labels[idx]


class WildBGModel(nn.Module):
    input_size: int = 202
    dataset: Dataset = WildBgDataSet
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(self.input_size, 300)
        self.fc2 = nn.Linear(300, 250)
        self.fc3 = nn.Linear(250, 200)
        self.output = nn.Linear(200, 6)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.output(x)
        return self.softmax(x)
