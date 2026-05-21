import torch
import tensorflow as tf
import numpy as np

def test_resize_images_divergence():
    """
    Adapted test case for tf.keras.backend.resize_images based on 
    PyTorch Issue 165105 (Matmul decomposition DDE).
    
    The original bug highlights an Eager/Compile divergence with float16 tensors.
    This test verifies that tf.keras.backend.resize_images produces consistent
    results between Eager execution and Graph (tf.function) execution using
    float16 data types and specific tensor dimensions found in the fuzzer output.
    """
    
    # Replicate the float16 context from the original bug report
    dtype = tf.float16
    
    # The original fuzzer produced tensors with shapes like (14, 416).
    # resize_images requires a 4D tensor. We adapt the dimensions to 
    # (Batch, Height, Width, Channels) -> (1, 14, 416, 3).
    # We use a fixed seed to ensure reproducibility, similar to torch.manual_seed(70609).
    tf.random.set_seed(70609)
    
    # Create input tensor on GPU if available, mimicking 'device=cuda'
    # Note: TensorFlow handles device placement automatically or via strategy.
    # We define the tensor here.
    x = tf.random.uniform((1, 14, 416, 3), minval=-1.0, maxval=1.0, dtype=dtype)

    # Define the operation using tf.keras.backend.resize_images
    # We use 'bilinear' interpolation as it is more computationally complex than 'nearest',
    # increasing the likelihood of exposing numerical or compilation divergences.
    def resize_op(input_tensor):
        return tf.keras.backend.resize_images(
            input_tensor,
            height_factor=2,
            width_factor=2,
            data_format='channels_last',
            interpolation='bilinear'
        )

    # 1. Execute in Eager Mode
    result_eager = resize_op(x)

    # 2. Execute in Compiled Mode (Graph Mode) - analogous to torch._dynamo/compile
    @tf.function
    def compiled_resize(input_tensor):
        return resize_op(input_tensor)

    result_compiled = compiled_resize(x)

    # 3. Verify Consistency (Check for Divergence)
    # Float16 operations can have slight numerical differences, but large divergences
    # indicate a bug similar to the one reported.
    diff = tf.abs(result_eager - result_compiled)
    max_diff = tf.reduce_max(diff).numpy()

    print(f"Max difference between Eager and Compiled: {max_diff}")

    # Tolerance of 1e-2 is chosen for float16 interpolation stability
    assert max_diff < 1e-2, (
        f"Divergence detected! Max diff: {max_diff}. "
        "This mirrors the Eager/Compile divergence in the original PyTorch bug."
    )
    
    print("Test Passed: No divergence detected.")

if __name__ == "__main__":
    test_resize_images_divergence()