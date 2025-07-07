import torch
import torch.nn as nn
import torch.onnx

class SimpleConvNet(nn.Module):
    """A simple CNN for creating a test ONNX model."""
    def __init__(self):
        super(SimpleConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x

def create_test_model(file_path: str = "sample_model.onnx"):
    """Creates and saves a sample ONNX model."""
    model = SimpleConvNet()
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32) # NCHW format

    torch.onnx.export(
        model,
        dummy_input,
        file_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=12,
        do_constant_folding=True,
    )
    print(f"Sample ONNX model saved to {file_path}")

if __name__ == "__main__":
    create_test_model() 