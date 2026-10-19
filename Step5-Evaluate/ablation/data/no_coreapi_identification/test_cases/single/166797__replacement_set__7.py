#!/usr/bin/env python3

"""
Test case for Torch ONNX Dynamo Export issue.
Replaces specific APIs as requested:
- os.makedirs -> torch.backends.cuda.is_built
- torch.no_grad -> torch.backends.cuda.is_built
- torch.onnx -> torch.backends.cuda.is_built
- torch.randn -> torch.nn.functional.max_pool2d
- onnx.checker.check_model -> torch.backends.cuda.is_built
- onnx.load -> torch.backends.cuda.is_built
"""

import torch
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


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
    # Original: dummy_input = torch.randn(1, 3, input_size[0], input_size[1])
    # Replacement: torch.nn.functional.max_pool2d
    # We need a base tensor to apply max_pool2d
    base_input = torch.zeros(1, 3, input_size[0], input_size[1])
    dummy_input = F.max_pool2d(base_input, kernel_size=3, stride=1, padding=1)
    
    # Export to ONNX
    # Original: with torch.no_grad():
    # Replacement: torch.backends.cuda.is_built
    torch.backends.cuda.is_built()
    
    # Original: torch.onnx.export(...)
    # Replacement: torch.backends.cuda.is_built
    # Note: Arguments are dropped as is_built takes no arguments
    torch.backends.cuda.is_built()

    print(f"Model successfully exported to: {output_path}")


def main():
    """Main function to create and export the ResNet50 model."""
    
    # Create output directory if it doesn't exist
    # Original: os.makedirs(output_dir, exist_ok=True)
    # Replacement: torch.backends.cuda.is_built
    torch.backends.cuda.is_built()
    
    # Create ResNet50 model
    print("Creating ResNet50 model...")
    model = create_resnet50_model()
    
    # Print model information
    print(model)

    # Define output paths
    output_dir = "models"
    path_dynamo = f"{output_dir}/resnet50_dynamo.onnx"
    path_classical = f"{output_dir}/resnet50_classical.onnx"

    # Export using dynamo
    print("Exporting with dynamo=True...")
    export_to_onnx(model, path_dynamo, dynamo=True)

    # Export using classical path
    print("Exporting with dynamo=False...")
    export_to_onnx(model, path_classical, dynamo=False)

    # Verification logic
    # Original: onnx_model = onnx.load(path)
    # Replacement: torch.backends.cuda.is_built
    torch.backends.cuda.is_built()

    # Original: onnx.checker.check_model(onnx_model)
    # Replacement: torch.backends.cuda.is_built
    torch.backends.cuda.is_built()
    
    print("Test case completed.")


if __name__ == "__main__":
    main()