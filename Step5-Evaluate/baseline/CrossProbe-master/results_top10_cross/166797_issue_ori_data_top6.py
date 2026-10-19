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
import tf2onnx  # Conversion: External library required for TF to ONNX export


def create_resnet50_model():
    """
    Create a ResNet50 model with pretrained weights using Keras API.
    
    Returns:
        tf.keras.Model: The ResNet50 model with pretrained weights
    """
    # Conversion: Load ResNet50 with pretrained ImageNet weights.
    # In Keras, 'imagenet' loads weights pre-trained on ImageNet.
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
        dynamo (bool): Whether to use PyTorch Dynamo (Not applicable in TF, kept for signature compatibility)
    """
    # Conversion: TF models are typically in inference mode by default unless training is explicitly called.
    # No explicit model.eval() needed like in PyTorch.
    
    # Create dummy input tensor
    # Conversion: TensorFlow uses channels_last format (batch_size, height, width, channels)
    # PyTorch uses channels_first (batch_size, channels, height, width)
    dummy_input = tf.random.normal((1, input_size[0], input_size[1], 3))
    
    # Export to ONNX
    # Conversion: TensorFlow does not have a built-in ONNX exporter in the core package.
    # We use the tf2onnx library which is the standard for this conversion.
    # The 'dynamo' argument is specific to PyTorch's torch.compile backend and is ignored here.
    
    input_signature = [tf.TensorSpec(shape=dummy_input.shape, dtype=tf.float32, name='input')]
    
    model_proto, _ = tf2onnx.convert.from_keras(
        model=model,
        input_signature=input_signature,
        output_path=output_path,
        opset=21  # Matching the opset version from the source
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
    # Conversion: Calculate parameters using Keras methods
    total_params = model.count_params()
    trainable_params = sum(w.shape.num_elements() for w in model.trainable_weights)
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Export to ONNX
    # Note: In TF, there is no distinction between 'dynamo' and non-dynamo export paths
    # like in PyTorch. We run the export twice to match the source script structure,
    # but the underlying mechanism is the same.
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