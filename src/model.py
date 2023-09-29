from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(in_features=202, out_features=300)
        self.fc2 = nn.Linear(in_features=300, out_features=250)
        self.fc3 = nn.Linear(in_features=250, out_features=200)
        self.fc4 = nn.Linear(in_features=200, out_features=6)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.softmax(x, dim=1)
