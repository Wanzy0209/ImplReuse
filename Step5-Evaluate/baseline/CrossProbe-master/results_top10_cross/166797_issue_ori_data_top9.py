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
import tf2onnx  # Conversion library for TF to ONNX


def create_resnet50_model():
    """
    Create a ResNet50 model with pretrained weights using Keras API.
    
    Returns:
        tf.keras.Model: The ResNet50 model with pretrained weights
    """
    # Load ResNet50 with pretrained ImageNet weights
    # Conversion: torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT) 
    # -> tf.keras.applications.ResNet50(weights='imagenet')
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
        dynamo (bool): Whether to use PyTorch Dynamo (No-op in TF, kept for structure)
    """
    # Set model to evaluation mode
    # Conversion: model.eval() is implicit in TF for inference/export, but we ensure no training ops
    
    # Create dummy input tensor
    # Shape: (batch_size, height, width, channels)
    # Conversion: PyTorch uses NCHW (1, 3, H, W), TensorFlow uses NHWC (1, H, W, 3)
    dummy_input = tf.random.normal((1, input_size[0], input_size[1], 3))
    
    # Export to ONNX
    # Conversion: torch.onnx.export -> tf2onnx.convert.from_keras
    # Note: tf2onnx handles the graph extraction. The 'dynamo' flag is PyTorch specific.
    # In TF, the graph is static by default for export.
    input_signature = [tf.TensorSpec(shape=(1, input_size[0], input_size[1], 3), dtype=tf.float32, name='input')]
    
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=21,  # Matching opset version
        output_path=output_path,
        output_names=['output']
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
    # Conversion: p.numel() -> tf.size(w).numpy()
    total_params = model.count_params()
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