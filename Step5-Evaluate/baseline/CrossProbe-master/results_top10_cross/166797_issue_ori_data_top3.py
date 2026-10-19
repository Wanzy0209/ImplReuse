```python
#!/usr/bin/env python3

"""
Script to create a TensorFlow ResNet50 model and export it to ONNX.
Uses tf.keras.applications.ResNet50 API.
"""

import tensorflow as tf
import os
import onnx
import tf2onnx  # Conversion note: Required for exporting TF models to ONNX


def create_resnet50_model():
    """
    Create a ResNet50 model with pretrained weights using Keras API.
    
    Returns:
        tf.keras.Model: The ResNet50 model with pretrained weights
    """
    # Load ResNet50 with pretrained ImageNet weights
    # Conversion note: weights=ResNet50_Weights.DEFAULT -> weights='imagenet'
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
        dynamo (bool): Whether to use PyTorch Dynamo (Ignored in TF, mapped to standard export)
    """
    # Conversion note: model.eval() is not strictly required in TF for inference,
    # but ensures layers like Dropout/BatchNorm behave correctly.
    # TF models are usually in inference mode by default unless training=True is specified.

    # Create dummy input tensor
    # Conversion note: PyTorch uses (N, C, H, W), TensorFlow uses (N, H, W, C).
    # We define the input signature for the exporter.
    # Shape: (batch_size, height, width, channels)
    dummy_input = tf.random.normal((1, input_size[0], input_size[1], 3))

    # Export to ONNX
    # Conversion note: torch.no_grad is not needed in TF.
    # Conversion note: torch.onnx.export is replaced by tf2onnx.convert.from_keras.
    # The 'dynamo' argument is specific to PyTorch 2.0 compiler; TF uses tf.function/XLA internally.
    # We perform the standard export here.
    
    input_signature = [tf.TensorSpec(shape=dummy_input.shape, dtype=tf.float32, name='input')]
    
    model_proto, _ = tf2onnx.convert.from_keras(
        model=model,
        input_signature=input_signature,
        output_path=output_path,
        opset=21  # Matching opset version from source
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
    # Conversion note: p.numel() -> tf.size(p).numpy() or similar logic
    total_params = model.count_params()
    # In Keras, all params are usually trainable unless explicitly frozen, 
    # but we can check trainable_variables if needed.
    # For ResNet50, count_params() returns total.
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Export to ONNX
    output_dynamo_path = os.path.join(output_dir, "resnet50_dynamo.onnx")
    print(f"Exporting model to ONNX...")
    print(f"Input image size: 224x224")

    # Conversion note: The 'dynamo' flag in PyTorch enables the new compiler.
    # In TensorFlow, graph optimization is handled differently (tf.function, XLA).
    # We pass the flag but the export logic remains the same (tf2onnx).
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