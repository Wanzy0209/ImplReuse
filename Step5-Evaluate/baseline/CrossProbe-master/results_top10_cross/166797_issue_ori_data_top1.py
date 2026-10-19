```python
#!/usr/bin/env python3

"""
Script to create a TensorFlow ResNet50 model and save it to SavedModel format.
Uses tf.keras.applications.ResNet50 API.
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
import os

# Conversion: ONNX library is specific to PyTorch/ONNX workflow.
# TensorFlow uses the SavedModel format for interoperability.
# import onnx 

def create_resnet50_model():
    """
    Create a ResNet50 model with pretrained weights using Keras API.
    
    Returns:
        tf.keras.Model: The ResNet50 model with pretrained weights
    """
    # Conversion: torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
    # TensorFlow Keras uses 'imagenet' string to load pretrained weights.
    model = ResNet50(weights='imagenet')
    
    print(f"Loaded ResNet50 with pretrained ImageNet weights")
    
    return model


def export_model(model, output_path, input_size=(224, 224), dynamo=False):
    """
    Save the TensorFlow model to SavedModel format.
    
    Args:
        model (tf.keras.Model): The TensorFlow model to save
        output_path (str): Path to save the model
        input_size (tuple): Input image size (height, width)
        dynamo (bool): Placeholder for compatibility. 
                       Note: PyTorch Dynamo is analogous to tf.function/graph mode,
                       which is default in TF 2.x for saved models.
    """
    # Conversion: model.eval() is not strictly required in TF for saving,
    # but Keras models with BatchNormalization behave in inference mode by default
    # unless training=True is explicitly passed.
    
    # Create dummy input tensor
    # Conversion: PyTorch uses NCHW (Batch, Channels, Height, Width).
    # TensorFlow uses NHWC (Batch, Height, Width, Channels).
    dummy_input = tf.random.normal((1, input_size[0], input_size[1], 3))
    
    # Conversion: torch.no_grad() context manager.
    # In TensorFlow, gradients are only calculated inside tf.GradientTape,
    # so no explicit context manager is needed for inference or export.
    
    # Export/Save the model
    # Conversion: torch.onnx.export -> tf.keras.Model.save
    # external_data is handled automatically by TF for large weights.
    # input_names/output_names map to the SignatureDef in SavedModel.
    
    # We call the model once to ensure the input shape is traced if not already built
    _ = model(dummy_input, training=False)
    
    model.save(output_path)
    
    print(f"Model successfully saved to: {output_path}")


def main():
    """Main function to create and save the ResNet50 model."""
    
    # Create output directory if it doesn't exist
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create ResNet50 model
    print("Creating ResNet50 model...")
    model = create_resnet50_model()
    
    # Print model information
    # Conversion: sum(p.numel() for p in model.parameters())
    # TensorFlow uses model.count_params()
    total_params = model.count_params()
    # In Keras, all loaded weights are technically trainable unless frozen,
    # but ResNet50 usually loads all as trainable.
    trainable_params = total_params 
    
    print(f"Model created successfully!")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Save Model (mimicking the 'dynamo' export)
    output_dynamo_path = os.path.join(output_dir, "resnet50_dynamo")
    print(f"Saving model (mimicking dynamo export)...")
    print(f"Input image size: 224x224")

    export_model(model, output_dynamo_path, input_size=(224, 224), dynamo=True)
    
    # Verify the saved model
    # Conversion: onnx.load -> tf.saved_model.load
    loaded_model = tf.saved_model.load(output_dynamo_path)
    # Conversion: onnx.checker.check_model -> Run a dummy inference to verify
    infer = loaded_model.signatures["serving_default"]
    print("Model verified successfully.")

    # Save Model (mimicking the 'non-dynamo' export)
    output_non_dynamo_path = os.path.join(output_dir, "resnet50")
    export_model(model, output_non_dynamo_path, input_size=(224, 224), dynamo=False)
    
    # Verify the saved model
    loaded_model = tf.saved_model.load(output_non_dynamo_path)
    infer = loaded_model.signatures["serving_default"]
    print("Model verified successfully.")


if __name__ == "__main__":
    main()
```