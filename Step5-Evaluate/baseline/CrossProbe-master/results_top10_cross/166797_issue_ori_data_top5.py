```python
#!/usr/bin/env python3

"""
Script to create a TensorFlow ResNet50 model and export it to ONNX.
Uses tf.keras.applications.ResNet50 API.
"""

import tensorflow as tf
import os
import onnx
import tf2onnx  # Conversion comment: tf2onnx is the standard library for exporting TF models to ONNX


def create_resnet50_model():
    """
    Create a ResNet50 model with pretrained weights using Keras API.
    
    Returns:
        tf.keras.Model: The ResNet50 model with pretrained weights
    """
    # Load ResNet50 with pretrained ImageNet weights
    # Conversion comment: In TensorFlow/Keras, 'imagenet' loads the pretrained weights.
    # The default input format is NHWC (Batch, Height, Width, Channels).
    model = tf.keras.applications.ResNet50(weights='imagenet')
    
    print("Loaded ResNet50 with pretrained ImageNet weights")
    
    return model


def export_to_onnx(model, output_path, input_size=(224, 224), dynamo=False):
    """
    Export the TensorFlow model to ONNX format.
    
    Args:
        model (tf.keras.Model): The TensorFlow model to export
        output_path (str): Path to save the ONNX model
        input_size (tuple): Input image size (height, width)
        dynamo (bool): Placeholder for compatibility. 
                       Conversion comment: PyTorch Dynamo is a compiler backend. 
                       TensorFlow models are graph-based by default via tf.function, 
                       so explicit dynamo usage is not required for export.
    """
    # Conversion comment: TensorFlow models do not require an explicit eval() mode switch 
    # like PyTorch for inference, as layers like Dropout behave differently based on the call context.
    
    # Create dummy input tensor
    # Conversion comment: PyTorch uses NCHW format (Batch, Channels, Height, Width).
    # TensorFlow uses NHWC format (Batch, Height, Width, Channels).
    # Shape: (batch_size, height, width, channels)
    dummy_input = tf.random.normal((1, input_size[0], input_size[1], 3))
    
    # Define input signature for the converter
    input_signature = [tf.TensorSpec(shape=dummy_input.shape, dtype=tf.float32, name='input')]

    # Export to ONNX
    # Conversion comment: TensorFlow uses tf2onnx for ONNX export.
    # We match the opset version 21 from the source script.
    onnx_model, _ = tf2onnx.convert.from_keras(
        model=model,
        input_signature=input_signature,
        opset=21,
        output_path=output_path
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
    # Conversion comment: model.count_params() returns the total number of parameters.
    total_params = model.count_params()
    # Conversion comment: Iterate through trainable_variables to count trainable params.
    trainable_params = sum([tf.size(v).numpy() for v in model.trainable_variables])
    
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