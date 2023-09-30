from . import Model
from torch import nn

class GnuBGModel(Model):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(202, 300)
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
