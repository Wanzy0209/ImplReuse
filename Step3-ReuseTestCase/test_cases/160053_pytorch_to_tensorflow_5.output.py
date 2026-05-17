import torch
import tensorflow as tf

def test_clip_by_norm_4d_input():
    """
    Adapted test case based on PyTorch Issue 160053.
    The original issue involved a 4D input causing an error in torch.nn.functional.pad.
    This test verifies that the similar TensorFlow API, tf.clip_by_norm, 
    handles 4D inputs correctly without raising dimension-related errors.
    """
    # Create a 4D tensor similar to the PyTorch bug report: torch.empty(2,2,2,2)
    # We use random values to ensure the norm calculation is meaningful.
    input_tensor = tf.random.uniform((2, 2, 2, 2), minval=0, maxval=10.0, dtype=tf.float32)
    
    # Define a clip norm value
    clip_norm = 2.0

    # Apply the similar API: tf.clip_by_norm
    # Original PyTorch call: F.pad(a, (1,1), mode="circular")
    # TensorFlow equivalent call (semantically adapted):
    try:
        result = tf.clip_by_norm(input_tensor, clip_norm)
    except Exception as e:
        print(f"Test Failed: API raised an exception for 4D input: {e}")
        raise

    # Verify the behavior
    # 1. Check that the shape is preserved (clipping does not change shape)
    assert result.shape == input_tensor.shape, \
        f"Shape mismatch: expected {input_tensor.shape}, got {result.shape}"

    # 2. Check that the L2-norm of the result is actually clipped
    # Calculate the global L2-norm of the resulting tensor
    l2_norm = tf.norm(result, ord='euclidean')
    
    # Allow for small floating point errors
    assert l2_norm.numpy() <= clip_norm + 1e-6, \
        f"Norm constraint failed: {l2_norm.numpy()} > {clip_norm}"

    print("Test passed: tf.clip_by_norm handles 4D input correctly.")

if __name__ == "__main__":
    test_clip_by_norm_4d_input()