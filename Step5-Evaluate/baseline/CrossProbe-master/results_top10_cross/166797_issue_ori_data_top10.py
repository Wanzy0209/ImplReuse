```python
#!/usr/bin/env python3

"""
Script to create a TensorFlow ResNet50 model and export it to SavedModel format.
Uses tf.keras.applications.ResNet50 API.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import os

# Conversion: onnx is not a native TensorFlow format, 
# so we use the native SavedModel format instead.

def create_resnet50_model():
    """
    Create a ResNet50 model with pretrained weights using tf.keras API.
    
    Returns:
        tf.keras.Model: The ResNet50 model with pretrained weights
    """
    # Load ResNet50 with pretrained ImageNet weights
    # Conversion: ResNet50_Weights.DEFAULT -> weights='imagenet'
    model = ResNet50(weights='imagenet')
    
    print(f"Loaded ResNet50 with pretrained ImageNet weights")
    
    return model


def export_to_savedmodel(model, output_path, input_size=(224, 224), dynamo=False):
    """
    Export the TensorFlow model to SavedModel format.
    
    Args:
        model (tf.keras.Model): The TensorFlow model to export
        output_path (str): Path to save the SavedModel
        input_size (tuple): Input image size (height, width)
        dynamo (bool): Whether to use PyTorch Dynamo (Ignored in TF, graph is default)
    """
    # Conversion: model.eval() is not needed in TF for saving, 
    # but we ensure the model is built.
    
    # Create dummy input tensor
    # Conversion: PyTorch uses (N, C, H, W), TensorFlow uses (N, H, W, C)
    dummy_input = tf.random.normal((1, input_size[0], input_size[1], 3))
    
    # Run a dummy pass to build the model layers if not already built
    _ = model(dummy_input, training=False)

    # Export to SavedModel
    # Conversion: torch.onnx.export -> model.save (SavedModel format)
    # Note: TensorFlow is graph-based by default, so 'dynamo' concept is native.
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
    # Conversion: p.numel() -> model.count_params()
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Export to SavedModel
    # Conversion: ONNX export -> SavedModel export
    output_dynamo_path = os.path.join(output_dir, "resnet50_dynamo")
    print(f"Exporting model to SavedModel...")
    print(f"Input image size: 224x224")

    export_to_savedmodel(model, output_dynamo_path, input_size=(224, 224), dynamo=True)
    
    # Verify the exported model
    # Conversion: onnx.load/check_model -> tf.saved_model.load
    loaded_model = tf.saved_model.load(output_dynamo_path)
    # Check if the default signature exists
    if "serving_default" in loaded_model.signatures:
        print("Model verified successfully.")
    else:
        print("Warning: serving_default signature not found.")

    output_non_dynamo_path = os.path.join(output_dir, "resnet50")
    export_to_savedmodel(model, output_non_dynamo_path, input_size=(224, 224), dynamo=False)
    
    # Verify the exported model
    loaded_model = tf.saved_model.load(output_non_dynamo_path)
    if "serving_default" in loaded_model.signatures:
        print("Model verified successfully.")
    else:
        print("Warning: serving_default signature not found.")


if __name__ == "__main__":
    main()
```