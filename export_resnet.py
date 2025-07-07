import torch
import torchvision.models as models

def export_resnet18():
    """
    Loads a pretrained ResNet-18 model from torchvision,
    and exports it to the ONNX format.
    """
    # Load a pretrained ResNet-18 model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()

    # Create a dummy input tensor that matches the model's expected input
    # Batch_size=1, 3 color channels, 224x224 image
    dummy_input = torch.randn(1, 3, 224, 224) 

    # Export the model to ONNX format
    torch.onnx.export(
        model, 
        dummy_input, 
        "resnet18.onnx", 
        export_params=True,
        opset_version=12,
        input_names=['input'],
        output_names=['output'],
        do_constant_folding=True
    )

    print("ResNet-18 model has been successfully exported to resnet18.onnx")

if __name__ == '__main__':
    export_resnet18() 