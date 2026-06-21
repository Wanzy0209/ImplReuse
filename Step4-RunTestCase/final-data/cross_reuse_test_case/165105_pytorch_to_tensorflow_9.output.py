import torch
import numpy as np

def test_tf_conv2d_transpose_divergence():
    """
    Adapted test case for tf.nn.conv2d_transpose based on PyTorch Matmul decomposition bug.
    The original bug involves float16 precision, specific constant values, and divergence
    between eager and compiled execution modes.
    """

    # Handle environment dependency issues (e.g., libstdc++ version mismatch)
    try:
        import tensorflow as tf
    except ImportError as e:
        print(f"Test skipped: TensorFlow import failed due to environment dependency issues (e.g., libstdc++ version). Error: {e}")
        return

    # Check for GPU availability to match the original bug's context (device=cuda)
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("Test skipped: No GPU available.")
        return

    # Reproduce the specific constants from the PyTorch fuzzer output
    # var_node_4 = torch.full((14,), 1.2255859375, dtype=torch.float16)
    # var_node_9 = torch.full((6, 13), 1.3154296875, dtype=torch.float16)
    const_val_1 = 1.2255859375
    const_val_2 = 1.3154296875
    const_val_3 = 0.1331787109375

    # Use float16 as in the original bug
    dtype = tf.float16

    # Map PyTorch Matmul dimensions to Conv2DTranspose dimensions
    # PyTorch: (14, 6) @ (6, 13) -> (14, 13)
    # TF Conv2DTranspose:
    # Input: (Batch, Height, Width, In_Channels) -> (1, 14, 6, 1)
    # Filter: (Kernel_H, Kernel_W, Out_Channels, In_Channels) -> (1, 1, 13, 1)
    # Output: (Batch, Height, Width, Out_Channels) -> (1, 14, 6, 13)

    input_shape = [1, 14, 6, 1]
    filter_shape = [1, 1, 13, 1]
    output_shape = [1, 14, 6, 13]

    # Create tensors using the fuzzer constants
    # Using tf.fill to mimic torch.full
    input_tensor = tf.fill(input_shape, tf.cast(const_val_1, dtype))
    kernel = tf.fill(filter_shape, tf.cast(const_val_2, dtype))

    # 1. Eager Execution
    print("Running Eager Execution...")
    try:
        eager_result = tf.nn.conv2d_transpose(
            input_tensor,
            kernel,
            output_shape=output_shape,
            strides=[1, 1, 1, 1],
            padding='VALID',
            data_format='NHWC'
        )
        print(f"Eager Result Shape: {eager_result.shape}")
        print(f"Eager Result Sample: {eager_result[0, 0, 0, :].numpy()}")
    except Exception as e:
        print(f"Eager Execution Failed: {e}")
        return

    # 2. Compiled Execution (tf.function) - mimics torch.compile
    print("\nRunning Compiled Execution (tf.function)...")
    @tf.function
    def compiled_conv2d_transpose(x, k):
        return tf.nn.conv2d_transpose(
            x,
            k,
            output_shape=output_shape,
            strides=[1, 1, 1, 1],
            padding='VALID',
            data_format='NHWC'
        )

    try:
        compiled_result = compiled_conv2d_transpose(input_tensor, kernel)
        print(f"Compiled Result Shape: {compiled_result.shape}")
        print(f"Compiled Result Sample: {compiled_result[0, 0, 0, :].numpy()}")
    except Exception as e:
        print(f"Compiled Execution Failed: {e}")
        return

    # 3. Verification / Divergence Check
    # The original bug was about "Eager/Compile Divergence".
    # We check if the results are identical or close.
    print("\nChecking for Divergence...")

    # Check for NaNs
    if tf.reduce_any(tf.math.is_nan(eager_result)):
        print("ERROR: NaNs found in Eager result")
    if tf.reduce_any(tf.math.is_nan(compiled_result)):
        print("ERROR: NaNs found in Compiled result")

    # Compare results
    # Note: float16 can have slight precision differences, but divergence usually means large errors or crashes.
    # We check if they are exactly equal first (common in simple deterministic ops), then close.
    are_equal = tf.reduce_all(tf.equal(eager_result, compiled_result))

    if are_equal:
        print("SUCCESS: Eager and Compiled results are identical.")
    else:
        diff = tf.abs(eager_result - compiled_result)
        max_diff = tf.reduce_max(diff)
        print(f"WARNING: Results differ. Max difference: {max_diff.numpy()}")

        # If difference is significant, it indicates a divergence bug
        if max_diff > 1e-3: # Threshold for float16
            print("FAILURE: Significant divergence detected between Eager and Compiled modes.")

if __name__ == "__main__":
    test_tf_conv2d_transpose_divergence()