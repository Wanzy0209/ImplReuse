```python
#!/usr/bin/env python3

"""
Script to create a TensorFlow ResNet50 model and export it to ONNX.
Uses tf.keras.applications.ResNet50 API.
"""

import tensorflow as tf
import os
import onnx
import tf2onnx  # Required for exporting TF models to ONNX


def create_resnet50_model():
    """
    Create a ResNet50 model with pretrained weights using Keras API.
    
    Returns:
        tf.keras.Model: The ResNet50 model with pretrained weights
    """
    # Load ResNet50 with pretrained ImageNet weights
    # Conversion: PyTorch uses weights object, TF uses string 'imagenet'
    model = tf.keras.applications.ResNet50(weights='imagenet')
    
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
    # Conversion: TF models are inference-ready by default, no explicit eval() needed
    # but we ensure the model is built.
    
    # Create dummy input tensor
    # Conversion: PyTorch uses NCHW format (batch, channels, height, width).
    # TensorFlow uses NHWC format (batch, height, width, channels).
    dummy_input = tf.random.normal((1, input_size[0], input_size[1], 3))
    
    # Define input signature for tracing
    input_signature = [tf.TensorSpec(shape=dummy_input.shape, dtype=tf.float32, name='input')]

    # Export to ONNX
    # Conversion: PyTorch uses torch.onnx.export. TensorFlow uses tf2onnx.
    # The 'dynamo' argument is specific to PyTorch's graph capture.
    # We use opset 21 to match the source script.
    model_proto, _ = tf2onnx.convert.from_keras(
        model=model,
        input_signature=input_signature,
        output_path=output_path,
        opset=21
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
    # Conversion: PyTorch uses p.numel(). TF uses count_params() or size().
    total_params = model.count_params()
    trainable_params = sum([w.shape.num_elements() for w in model.trainable_weights])
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Export to ONNX
    output_dynamo_path = os.path.join(output_dir, "resnet50_dynamo.onnx")
    print(f"Exporting model to ONNX...")
    print(f"Input image size: 224x224")

    # Note: The 'dynamo' flag in the original script controls PyTorch's internal graph capture.
    # In TensorFlow, tf2onnx handles the conversion. We call it twice to preserve the test structure.
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