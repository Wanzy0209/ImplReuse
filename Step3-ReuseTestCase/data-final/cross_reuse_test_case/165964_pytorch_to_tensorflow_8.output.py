import torch
import tensorflow as tf

# Check for GPU availability to match the context of the original bug report
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"Testing on GPU: {gpus[0].name}")
    
    try:
        # Replicate the tensor creation part of the original test case
        # torch.ones(1, device="cuda") -> tf.ones([1]) (placed on GPU automatically)
        x = tf.ones([1])
        
        # Apply the similar API: tf.keras.ops.negative
        # This tests the behavior of the API on a GPU-allocated tensor
        result = tf.keras.ops.negative(x)
        
        # Verify the result (mimicking .item() in the original test case)
        print(f"Result: {result.numpy()}")
        
        # Assertion to ensure the operation worked as expected
        assert result.numpy()[0] == -1.0, "Expected -1.0"
        print("Test passed: tf.keras.ops.negative executed successfully on GPU.")
        
    except tf.errors.ResourceExhaustedError as e:
        # Catching potential OOM errors similar to the PyTorch bug
        print(f"Test failed with ResourceExhaustedError (OOM): {e}")
    except Exception as e:
        print(f"Test failed with unexpected error: {e}")
else:
    print("No GPU available. Skipping test.")