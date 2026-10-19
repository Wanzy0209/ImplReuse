```python
#!/usr/bin/env python3

"""
Script to create a TensorFlow ResNet50 model and export it to ONNX.
Uses tf.keras.applications.ResNet50 API.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import os
import onnx
# Conversion comment: PyTorch uses torch.onnx.export, TensorFlow uses tf2onnx
import tf2onnx

def create_resnet50_model():
    """
    Create a ResNet50 model with pretrained weights using Keras API.
    
    Returns:
        tf.keras.Model: The ResNet50 model with pretrained weights
    """
    # Conversion comment: Load ResNet50 with pretrained ImageNet weights using Keras
    model = ResNet50(weights='imagenet')
    
    print(f"Loaded ResNet50 with pretrained ImageNet weights")
    
    return model


def export_to_onnx(model, output_path, input_size=(224, 224), dynamo=False):
    """
    Export the TensorFlow model to ONNX format.
    
    Args:
        model (tf.keras.Model): The TensorFlow model to export
        output_path (str): Path to save the ONNX model
        input_size (tuple): Input image size (height, width)
        dynamo (bool): Whether to use PyTorch Dynamo (Unused in TF, kept for signature compatibility)
    """
    # Conversion comment: TF models are typically in NHWC format (Batch, Height, Width, Channels)
    # Conversion comment: Define input signature for conversion
    input_signature = [tf.TensorSpec(shape=(1, input_size[0], input_size[1], 3), dtype=tf.float32, name='input')]
    
    # Export to ONNX
    # Conversion comment: Using tf2onnx.convert.from_keras as TF equivalent to torch.onnx.export
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        output_path=output_path,
        opset=21, # Matching opset version
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
    # Conversion comment: Keras models use count_params() instead of sum(p.numel())
    total_params = model.count_params()
    # Conversion comment: Calculate trainable params by summing sizes of trainable weights
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
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
```