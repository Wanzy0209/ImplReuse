#!/usr/bin/env python3

"""
Script to create a PyTorch ResNet50 model and export it to ONNX.
Uses torchvision.models.resnet.ResNet API.
"""

import torch
import torch.onnx
from torchvision.models import resnet50, ResNet50_Weights
import os

import onnx


def create_resnet50_model():
    """
    Create a ResNet50 model with pretrained weights using torchvision API.
    
    Returns:
        torch.nn.Module: The ResNet50 model with pretrained weights
    """
    # Load ResNet50 with pretrained ImageNet weights using the new API
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    
    print(f"Loaded ResNet50 with pretrained ImageNet weights: {weights}")
    
    return model


def export_to_onnx(model, output_path, input_size=(224, 224), dynamo=False):
    """
    Export the PyTorch model to ONNX format.
    
    Args:
        model (torch.nn.Module): The PyTorch model to export
        output_path (str): Path to save the ONNX model
        input_size (tuple): Input image size (height, width)
        dynamo (bool): Whether to use PyTorch Dynamo
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input tensor
    # Shape: (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            model=model,
            args=dummy_input,
            f=output_path,
            external_data=True,
            export_params=True,
            opset_version=21,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            verbose=True,
            dynamo=dynamo,
            fallback=False,
        )

    print(f"Model successfully exported to: {output_path}")


def main():
    """Main function to create and export the ResNet50 model."""
    
    # Create output directory if it doesn't exist
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create ResNet50 model
    print("Creating ResNet50 model...")
    model = create_resnet50_model()
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Export to ONNX
    output_dynamo_path = os.path.join(output_dir, "resnet50_dynamo.onnx")
    print(f"Exporting model to ONNX...")
    print(f"Input image size: 224x224")

    export_to_onnx(model, output_dynamo_path, input_size=(224, 224), dynamo=True)
    # Verify the exported model
    onnx_dynamo_model = onnx.load(output_dynamo_path)
    onnx.checker.check_model(onnx_dynamo_model)

    output_non_dynamo_path = os.path.join(output_dir, "resnet50.onnx")
    export_to_onnx(model, output_non_dynamo_path, input_size=(224, 224), dynamo=False)
    # Verify the exported model
    onnx_non_dynamo_model = onnx.load(output_non_dynamo_path)
    onnx.checker.check_model(onnx_non_dynamo_model)


if __name__ == "__main__":
    main()