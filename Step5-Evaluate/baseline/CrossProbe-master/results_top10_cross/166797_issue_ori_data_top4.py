```python
#!/usr/bin/env python3

"""
Script to create a TensorFlow ResNet50 model and export it to SavedModel format.
Uses tf.keras.applications.ResNet50 API.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import os
import numpy as np


def create_resnet50_model():
    """
    Create a ResNet50 model with pretrained weights using Keras API.
    
    Returns:
        tf.keras.Model: The ResNet50 model with pretrained weights
    """
    # Conversion: Load ResNet50 with pretrained ImageNet weights
    # PyTorch: weights = ResNet50_Weights.DEFAULT; model = resnet50(weights=weights)
    # TensorFlow: weights='imagenet'
    model = ResNet50(weights='imagenet')
    
    print(f"Loaded ResNet50 with pretrained ImageNet weights")
    
    return model


def export_to_onnx(model, output_path, input_size=(224, 224), dynamo=False):
    """
    Export the TensorFlow model to SavedModel format.
    
    Args:
        model (tf.keras.Model): The TensorFlow model to export
        output_path (str): Path to save the SavedModel
        input_size (tuple): Input image size (height, width)
        dynamo (bool): Whether to use explicit tf.function tracing (analogous to PyTorch Dynamo)
    """
    # Conversion: model.eval() is not strictly required in TF for saving
    # but we ensure the model is in inference mode if we were to run data.
    
    # Create dummy input tensor
    # Conversion: PyTorch uses NCHW (batch, channels, height, width)
    # TensorFlow uses NHWC (batch, height, width, channels)
    # Shape: (batch_size, height, width, channels)
    dummy_input = tf.random.normal((1, input_size[0], input_size[1], 3))
    
    # Export to SavedModel
    # Conversion: torch.no_grad() is not needed in TF for inference/export
    # Conversion: torch.onnx.export -> model.save (SavedModel format)
    # Note: TensorFlow does not support native ONNX export. 
    # We export to the standard SavedModel format.
    
    if dynamo:
        # Conversion: dynamo=True in PyTorch uses torch.compile.
        # In TensorFlow, we use tf.function to trace the graph explicitly.
        @tf.function
        def call_model(x):
            return model(x, training=False)
        
        # Warm up the model to ensure graph is traced
        _ = call_model(dummy_input)
        # Note: We save the original Keras model, which includes the graph structure.
    
    # Save the model
    # Conversion: external_data, export_params, opset_version are ONNX specific.
    # We use the standard Keras save method.
    model.save(output_path)

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
    # Conversion: p.numel() -> model.count_params() or tf.size(w)
    total_params = model.count_params()
    trainable_params = sum([np.prod(w.shape) for w in model.trainable_weights])
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Export
    # Conversion: .onnx extension removed for SavedModel directory
    output_dynamo_path = os.path.join(output_dir, "resnet50_dynamo")
    print(f"Exporting model...")
    print(f"Input image size: 224x224")

    export_to_onnx(model, output_dynamo_path, input_size=(224, 224), dynamo=True)
    
    # Verify the exported model
    # Conversion: onnx.load -> tf.saved_model.load
    # Conversion: onnx.checker.check_model -> running inference to verify
    loaded_model = tf.saved_model.load(output_dynamo_path)
    infer = loaded_model.signatures["serving_default"]
    dummy_input = tf.random.normal((1, 224, 224, 3))
    output = infer(dummy_input)
    print(f"Verification successful for dynamo model. Output keys: {list(output.keys())}")

    output_non_dynamo_path = os.path.join(output_dir, "resnet50")
    export_to_onnx(model, output_non_dynamo_path, input_size=(224, 224), dynamo=False)
    
    # Verify the exported model
    loaded_model = tf.saved_model.load(output_non_dynamo_path)
    infer = loaded_model.signatures["serving_default"]
    dummy_input = tf.random.normal((1, 224, 224, 3))
    output = infer(dummy_input)
    print(f"Verification successful for standard model. Output keys: {list(output.keys())}")


if __name__ == "__main__":
    main()
```