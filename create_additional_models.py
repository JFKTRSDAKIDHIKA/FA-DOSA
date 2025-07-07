#!/usr/bin/env python3
"""
Script to create additional ONNX models for multi-workload experiments.
Generates various neural network architectures to test FA-DOSA on diverse workloads.
"""

import torch
import torch.nn as nn
import torchvision.models as models

def create_mobilenet_v2():
    """Create and export MobileNet-V2 model."""
    print("Creating MobileNet-V2...")
    model = models.mobilenet_v2(pretrained=False)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    torch.onnx.export(
        model,
        dummy_input,
        "mobilenet_v2.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("  ✓ mobilenet_v2.onnx created")

def create_simple_cnn():
    """Create and export a simple CNN model."""
    print("Creating Simple CNN...")
    
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2, 2)
            
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2, 2)
            
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
            self.relu3 = nn.ReLU()
            self.pool3 = nn.MaxPool2d(2, 2)
            
            self.global_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(128, 10)
        
        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.pool3(self.relu3(self.conv3(x)))
            x = self.global_pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x
    
    model = SimpleCNN()
    model.eval()
    
    dummy_input = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        model,
        dummy_input,
        "simple_cnn.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("  ✓ simple_cnn.onnx created")

def create_vgg_small():
    """Create and export a small VGG-style model."""
    print("Creating VGG-Small...")
    
    class VGGSmall(nn.Module):
        def __init__(self):
            super(VGGSmall, self).__init__()
            self.features = nn.Sequential(
                # Block 1
                nn.Conv2d(3, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Block 2
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                
                # Block 3
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(128, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 100)
            )
        
        def forward(self, x):
            x = self.features(x)
            x = self.classifier(x)
            return x
    
    model = VGGSmall()
    model.eval()
    
    dummy_input = torch.randn(1, 3, 64, 64)
    torch.onnx.export(
        model,
        dummy_input,
        "vgg_small.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("  ✓ vgg_small.onnx created")

def create_efficientnet_b0():
    """Create and export EfficientNet-B0 model (if available)."""
    try:
        print("Creating EfficientNet-B0...")
        # Try to use torchvision's EfficientNet (requires newer version)
        model = models.efficientnet_b0(pretrained=False)
        model.eval()
        
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model,
            dummy_input,
            "efficientnet_b0.onnx",
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print("  ✓ efficientnet_b0.onnx created")
    except AttributeError:
        print("  ⚠ EfficientNet not available in this torchvision version, skipping")

def main():
    """Generate all additional models."""
    print("Generating additional ONNX models for multi-workload experiments...")
    print("=" * 60)
    
    try:
        create_simple_cnn()
        create_vgg_small()
        create_mobilenet_v2()
        create_efficientnet_b0()
        
        print("\n" + "=" * 60)
        print("Model generation completed!")
        print("\nGenerated models:")
        
        import os
        models_created = []
        for model_file in ['simple_cnn.onnx', 'vgg_small.onnx', 'mobilenet_v2.onnx', 'efficientnet_b0.onnx']:
            if os.path.exists(model_file):
                size_mb = os.path.getsize(model_file) / (1024 * 1024)
                models_created.append(f"  - {model_file} ({size_mb:.1f} MB)")
        
        for model in models_created:
            print(model)
            
        if models_created:
            print(f"\nYou can now update the workloads list in run_experiments.py:")
            print("workloads = [")
            print("    'resnet18.onnx',")
            for model in models_created:
                model_name = model.split()[1]
                print(f"    '{model_name}',")
            print("]")
        
    except Exception as e:
        print(f"Error generating models: {e}")
        print("Make sure you have PyTorch and torchvision installed.")

if __name__ == "__main__":
    main() 