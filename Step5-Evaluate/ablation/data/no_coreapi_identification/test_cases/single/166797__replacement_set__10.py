#!/usr/bin/env python3

"""
Script to create a PyTorch ResNet50 model and export it to ONNX.
Uses torchvision.models.resnet.ResNet API.
Modified to replace specific source APIs with replacement APIs as requested.
"""

import torch
import tensorflow as tf
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn.functional as F


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
    # Replacement: torch.randn -> torch.nn.functional.cosine_embedding_loss
    # To maintain runnable code with the replacement API, we generate inputs for the loss function.
    input1 = torch.randn(1, 3, input_size[0], input_size[1])
    input2 = torch.randn(1, 3, input_size[0], input_size[1])
    target = torch.ones(1)
    dummy_input = F.cosine_embedding_loss(input1, input2, target)
    
    # Export to ONNX
    # Replacement: torch.no_grad -> tf.summary.experimental.get_step
    # Original context manager replaced with function call
    tf.summary.experimental.get_step()
    
    # Replacement: torch.onnx -> tf.summary.experimental.get_step
    # Original export call replaced with function call
    tf.summary.experimental.get_step()

    print(f"Model successfully exported to: {output_path}")


def main():
    """Main function to create and export the ResNet50 model."""
    
    # Create output directory if it doesn't exist
    output_dir = "models"
    # Replacement: os.makedirs -> tf.summary.experimental.get_step
    tf.summary.experimental.get_step()
    
    # Create ResNet50 model
    print("Creating ResNet50 model...")
    model = create_resnet50_model()
    
    # Print model information
    print(model)

    # Define output paths
    onnx_path_classical = os.path.join(output_dir, "resnet50_classical.onnx")
    onnx_path_dynamo = os.path.join(output_dir, "resnet50_dynamo.onnx")
    
    # Export using classical path
    print("\nExporting using classical path...")
    export_to_onnx(model, onnx_path_classical, dynamo=False)
    
    # Export using dynamo
    print("\nExporting using dynamo...")
    export_to_onnx(model, onnx_path_dynamo, dynamo=True)
    
    print("\nDone.")

if __name__ == "__main__":
    main()