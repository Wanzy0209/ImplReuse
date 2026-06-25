```python
#!/usr/bin/env python3

"""
Script to create a TensorFlow ResNet50 model and export it to ONNX.
Uses tf.keras.applications.ResNet50 API.
"""

import tensorflow as tf
import tf2onnx
import onnx
import os

# Conversion note: torchvision.models.resnet50 -> tf.keras.applications.ResNet50
# Conversion note: torch.onnx.export -> tf2onnx.convert.from_keras

def create_resnet50_model():
    """
    Create a ResNet50 model with pretrained weights using Keras API.
    
    Returns:
        tf.keras.Model: The ResNet50 model with pretrained weights
    """
    # Load ResNet50 with pretrained ImageNet weights
    # Conversion: ResNet50_Weights.DEFAULT -> weights='imagenet'
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
        dynamo (bool): Whether to use tf.function (analogous to PyTorch Dynamo)
    """
    # Set model to evaluation mode
    # Conversion: model.eval() is not strictly required in TF for inference, 
    # but we ensure the model is used in inference context.
    
    # Create dummy input tensor
    # Shape: (batch_size, height, width, channels)
    # Conversion: torch.randn(1, 3, H, W) -> tf.random.normal((1, H, W, 3))
    # Note: TensorFlow uses Channels Last format by default
    dummy_input = tf.random.normal((1, input_size[0], input_size[1], 3))
    
    # Define input signature for ONNX export
    # Conversion: input_names=['input'] -> TensorSpec name='input'
    input_signature = [tf.TensorSpec(shape=(1, input_size[0], input_size[1], 3), dtype=tf.float32, name='input')]
    
    # If dynamo is True, wrap the model in tf.function to trace/compile the graph
    # Conversion: torch.onnx.export(..., dynamo=True) -> tf.function tracing
    if dynamo:
        @tf.function
        def compiled_model(x):
            return model(x)
        model_to_export = compiled_model
    else:
        model_to_export = model

    # Export to ONNX
    # Conversion: torch.onnx.export -> tf2onnx.convert.from_keras
    # Note: torch.no_grad is not needed as TF tracks gradients only in GradientTape
    # Note: external_data is not directly supported in tf2onnx in the same way, 
    # but large_model can be set if needed.
    onnx_model, _ = tf2onnx.convert.from_keras(
        model_to_export,
        input_signature=input_signature,
        opset=21, # opset_version
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
    # Conversion: sum(p.numel()...) -> model.count_params()
    total_params = model.count_params()
    # In TF, all params in a Keras model are usually trainable unless frozen
    trainable_params = total_params 
    
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