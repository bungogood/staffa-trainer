from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from model import GnuBGModel, WildBGModel

def save_model(model: nn.Module, path: str) -> None:
    dummy_input = torch.randn(1, model.input_size, requires_grad=True)
    torch.onnx.export(model, dummy_input, path)

def train(model: nn.Module, trainloader: DataLoader, epochs: int) -> nn.Module:
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

def main(model: nn.Module, data_path: str, model_path: str):
    traindata = model.dataset(data_path).to(device)
    trainloader = DataLoader(traindata, batch_size=64, shuffle=True)

    try:
        model = train(model, trainloader, 20)
    finally:
        print('Finished Training')
        save_model(model, model_path)

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = "mps"
    else:
        device = "cpu"
    device = torch.device(device)
    print(f"Using {device} device")
    Path("../model").mkdir(exist_ok=True)
    model = WildBGModel().to(device)
    main(model, "data/rollouts.csv", "model/staffa.onnx")
