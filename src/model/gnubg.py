from torch.utils.data import Dataset
from torch import nn
import pandas as pd
import torch
import codecs

class GnuBGDataSet(Dataset):
    def __init__(self, csv_files: list | str):
        if isinstance(csv_files, str):
            csv_files = [csv_files]
        data = pd.concat([pd.read_csv(f) for f in csv_files ], ignore_index=True)
        self.inputs = torch.Tensor(data.iloc[:, 5:].values)
        self.labels = torch.Tensor(data.iloc[:, :5].values)
    
    def to(self, device):
        self.inputs = self.inputs.to(device)
        self.labels = self.labels.to(device)
        return self

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]



class GnuBGModel(nn.Module):
    input_size: int = 202
    dataset: Dataset = GnuBGDataSet
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(self.input_size, 300)
        self.fc2 = nn.Linear(300, 250)
        self.fc3 = nn.Linear(250, 200)
        self.output = nn.Linear(200, 5)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.output(x)
        return self.softmax(x)

def to_posid(pos: str) -> str:
    pid = "".join(hex(ord(c) - ord("A"))[2:] for c in pos)
    b64 = codecs.encode(codecs.decode(pid, 'hex'), 'base64').decode().strip()
    return b64[:-2]

def main():
    output = []
    with open("data/contact-train-data.txt", 'r') as f:
        positions = f.readlines()
        for line in positions:
            if line.startswith('#'):
                continue
            line = line.strip()
            line = line.split(' ')
            line[0] = to_posid(line[0])
            output.append(",".join(line))
    with open("data/contact-train-data.csv", 'w') as f:
        f.write("\n".join(output))

if __name__ == "__main__":
    main()
