from . import Model
from torch import nn
import codecs

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
