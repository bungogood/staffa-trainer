from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from model import ModelDataset, Model
from model.gnubg import GnuBGModel
from model.wildbg import WildBGModel
from model.onnx import ONNXModel

def train(model: Model, trainloader: DataLoader, epochs: int) -> Model:
    # Define loss function, L1Loss and MSELoss are good choices
    criterion = nn.MSELoss()

    # Optimizer based on model, adjust the learning rate
    # 4.0 has worked well for Tanh(), one layer and 100k positions
    # 3.0 has worked well for ReLu(), three layers and 200k positions
    optimizer = torch.optim.SGD(model.parameters(), lr=3.0)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            # set optimizer to zero grad to remove previous epoch gradients
            optimizer.zero_grad()
            # forward propagation
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            # backward propagation
            loss.backward()
            # optimize
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(trainloader) / 64
        print(f'[Epoch: {epoch + 1}] loss: {epoch_loss:.5f}')
    
    return model

def main(model: Model, traindata: Dataset, model_path: str):
    trainloader = DataLoader(traindata, batch_size=64, shuffle=True)

    try:
        model = train(model, trainloader, 20)
    finally:
        print('Finished Training')
        model.save_onnx(model_path)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    device = torch.device(device)
    csv_file = "data/rollouts-07.csv"
    print(f"Using {device} device")
    Path("../model").mkdir(exist_ok=True)
    model = WildBGModel().to(device)
    traindata = ModelDataset(model, csv_file, sep=";").to(device)
    main(model, traindata, "model/staffa.onnx")
