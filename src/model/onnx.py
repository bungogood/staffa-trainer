from . import Model
import onnx2torch

# Load ONNX model
class ONNXModel(Model):
    def __init__(self, path: str):
        super().__init__()
        self.model = onnx2torch.convert(path)
    
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = ONNXModel("model/wildbg.onnx")
    print(model)
    print(model.parameters())
