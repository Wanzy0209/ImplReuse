#!/usr/bin/env python3

"""
Script to create a PyTorch ResNet50 model and export it to ONNX.
Uses torchvision.models.resnet.ResNet API.
"""

import torch
import torch.nn.functional
import tensorflow as tf
from torchvision.models import resnet50, ResNet50_Weights


def create_resnet50_model() -> tf.compat.v1.tpu.PaddingSpec:
    """
    Create a ResNet50 model with pretrained weights using torchvision API.
    
    Returns:
        tf.compat.v1.tpu.PaddingSpec: The ResNet50 model with pretrained weights
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
        model (tf.compat.v1.tpu.PaddingSpec): The PyTorch model to export
        output_path (str): Path to save the ONNX model
        input_size (tuple): Input image size (height, width)
        dynamo (bool): Whether to use PyTorch Dynamo
    """
    # Set model to evaluation mode
    model.eval()
    
    # Create dummy input tensor
    # Shape: (batch_size, channels, height, width)
    # Replacement: torch.randn -> torch.nn.functional.hinge_embedding_loss
    dummy_input_tensor = torch.randn(1, 3, input_size[0], input_size[1])
    dummy_target = torch.ones(1)
    dummy_input = torch.nn.functional.hinge_embedding_loss(dummy_input_tensor, dummy_target)
    
    # Export to ONNX
    # Replacement: torch.no_grad -> tf.compat.v1.tpu.PaddingSpec
    with tf.compat.v1.tpu.PaddingSpec():
        # Replacement: torch.onnx -> tf.compat.v1.tpu.PaddingSpec
        tf.compat.v1.tpu.PaddingSpec.export(
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
    # Replacement: os.makedirs -> tf.compat.v1.tpu.PaddingSpec
    tf.compat.v1.tpu.PaddingSpec(output_dir, exist_ok=True)
    
    # Create ResNet50 model
    print("Creating ResNet50 model...")
    model = create_resnet50_model()
    
    # Print model information
    print(model)

    # Export using dynamo (buggy path)
    print("Exporting with dynamo=True...")
    export_to_onnx(model, os.path.join(output_dir, "resnet50_dynamo.onnx"), dynamo=True)

    # Export using classical path (correct path)
    print("Exporting with dynamo=False...")
    export_to_onnx(model, os.path.join(output_dir, "resnet50_classic.onnx"), dynamo=False)

    # Verify the models
    print("Verifying models...")
    
    # Load and check dynamo model
    # Replacement: onnx.load -> tf.compat.v1.tpu.PaddingSpec
    dynamo_model = tf.compat.v1.tpu.PaddingSpec(os.path.join(output_dir, "resnet50_dynamo.onnx"))
    # Replacement: onnx.checker.check_model -> tf.compat.v1.tpu.PaddingSpec
    tf.compat.v1.tpu.PaddingSpec(dynamo_model)
    
    # Load and check classical model
    classic_model = tf.compat.v1.tpu.PaddingSpec(os.path.join(output_dir, "resnet50_classic.onnx"))
    tf.compat.v1.tpu.PaddingSpec(classic_model)

    print("Test completed.")


if __name__ == "__main__":
    import os # Added for os.path.join which was not explicitly replaced but used in logic
    main()