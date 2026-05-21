import torch
import tensorflow as tf
import numpy as np

def test_mobilenet_v2_numerics_and_consistency():
    """
    Adapts the PyTorch torch.export.export bug reproduction to TensorFlow.
    
    Original Bug: The exported program produced different results than the original model.
    Adaptation: 
    1. Use tf.debugging.enable_check_numerics to monitor for numerical instability (NaN/Inf).
    2. Compare the output of the eager model vs. the traced model (tf.function), 
       which is the semantic equivalent of torch.export.export.
    """
    
    # 1. Load Model
    # Using MobileNetV2 with random weights to match the PyTorch setup (weights=None)
    model = tf.keras.applications.MobileNetV2(weights=None)
    
    # 2. Prepare Input
    # PyTorch uses NCHW (1, 3, 224, 224), TensorFlow defaults to NHWC (1, 224, 224, 3)
    x = np.random.rand(1, 224, 224, 3).astype(np.float32)
    
    # 3. Enable Numerics Checking
    # This is the specific API requested. It will raise an error if any op produces NaN or Inf.
    tf.debugging.enable_check_numerics()
    
    # 4. Execute Eagerly (Original Model)
    y_eager = model(x)
    
    # 5. Execute Traced (Exported Model equivalent)
    # tf.function traces the model, similar to torch.export.export
    traced_model = tf.function(model)
    y_traced = traced_model(x)
    
    # 6. Verify Consistency
    # Check if the traced model produces the same results as the eager model.
    # If enable_check_numerics detected bad values, execution would have already failed.
    tf.debugging.assert_near(
        y_eager, 
        y_traced, 
        message="Mismatched elements: Traced model output differs from Eager model output"
    )
    
    print("Test passed: Numerics are valid and Eager/Traced outputs are consistent.")

if __name__ == "__main__":
    test_mobilenet_v2_numerics_and_consistency()