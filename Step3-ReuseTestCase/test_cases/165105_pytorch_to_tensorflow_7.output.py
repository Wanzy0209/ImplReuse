import torch
import tensorflow as tf
import numpy as np

def test_conv2d_transpose_divergence():
    """
    Adapted test case for tf.compat.v1.nn.conv2d_transpose based on 
    PyTorch matmul decomposition divergence bug (Issue 165105).
    
    The original bug involves eager/compile divergence with float16 matmuls.
    This test maps the matmul operation to a 1x1 conv2d_transpose to verify
    similar backend behavior in TensorFlow.
    """
    
    # Set seed to match the fuzzer report context
    tf.random.set_seed(70609)

    # --- Reproduce PyTorch var_node_9 and var_node_10 ---
    # PyTorch: var_node_9 = torch.full((6, 13), 1.3154296875, dtype=torch.float16)
    # PyTorch: var_node_10 = arg_1 (size=(13, 1))
    
    # Mapping to conv2d_transpose:
    # Input (var_node_9): (Batch=6, H=1, W=1, Channels=13)
    # Filter (var_node_10): (H=1, W=1, Output_Channels=1, Input_Channels=13)
    # Operation: (6, 13) @ (13, 1) -> (6, 1)
    
    input_val = 1.3154296875
    # Create input tensor (6, 1, 1, 13) filled with the specific float16 value
    input_tensor = tf.fill([6, 1, 1, 13], tf.cast(input_val, tf.float16))
    
    # Create filter tensor (1, 1, 1, 13) with random values to simulate fuzzer argument
    # Note: In conv2d_transpose, filter shape is [height, width, out_channels, in_channels]
    filter_tensor = tf.random.uniform([1, 1, 1, 13], minval=-1.0, maxval=1.0, dtype=tf.float16)

    # Define the operation logic
    def model_logic(x, f):
        return tf.compat.v1.nn.conv2d_transpose(
            x,
            f,
            output_shape=tf.shape(x)[:3] + [1], # Output (6, 1, 1, 1)
            strides=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC'
        )

    # 1. Run in Eager Mode
    eager_result = model_logic(input_tensor, filter_tensor)

    # 2. Run in Compiled Mode (Graph Mode) to check for divergence
    @tf.function
    def compiled_model(x, f):
        return model_logic(x, f)

    compiled_result = compiled_model(input_tensor, filter_tensor)

    # 3. Verify Consistency
    # Check if results are close enough. Significant divergence indicates a bug.
    # We use numpy for the assertion to handle the comparison clearly.
    diff = np.abs(eager_result.numpy() - compiled_result.numpy())
    
    # Allow for a small epsilon (1e-3) due to potential float16 precision differences
    # in different execution paths, but flag large divergences.
    if not np.all(diff < 1e-3):
        print("Eager Result Sample:", eager_result.numpy().flatten()[:5])
        print("Compiled Result Sample:", compiled_result.numpy().flatten()[:5])
        raise AssertionError(f"Eager/Compile divergence detected! Max diff: {np.max(diff)}")

    print("Test passed: No divergence between Eager and Compiled modes.")

if __name__ == "__main__":
    test_conv2d_transpose_divergence()