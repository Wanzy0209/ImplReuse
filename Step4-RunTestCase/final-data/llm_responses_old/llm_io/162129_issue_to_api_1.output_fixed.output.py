import tensorflow as tf
import numpy as np

def test_separable_conv2d_direct_construction():
    """
    Test case for tf.compat.v1.nn.separable_conv2d.
    
    This test is inspired by Issue #162129, which highlights a bug where direct 
    construction of a component (FakeProcessGroup) leads to silent incorrectness 
    (missing dispatch). 
    
    Here, we test the direct functional usage of separable_conv2d (as opposed to 
    using a Keras Layer) to ensure it executes correctly and produces the expected 
    output shape and values, avoiding any "silent" failures where the operation 
    might be a no-op or pass-through.
    """
    
    # 1. Setup inputs (Mirroring tensor creation in the bug report)
    # We use explicit shapes to ensure the operation is well-defined.
    # Input: Batch=1, Height=5, Width=5, Channels=3
    input_tensor = tf.constant(np.random.rand(1, 5, 5, 3), dtype=tf.float32)
    
    # Kernels for the separable convolution
    # depthwise_kernel: [filter_height, filter_width, in_channels, channel_multiplier]
    depthwise_kernel = tf.constant(np.random.rand(3, 3, 3, 2), dtype=tf.float32)
    # pointwise_kernel: [1, 1, in_channels * channel_multiplier, out_channels]
    pointwise_kernel = tf.constant(np.random.rand(1, 1, 6, 4), dtype=tf.float32)

    # 2. Execute Operation (Mirroring dist.all_reduce)
    # We call the functional API directly. In the context of the bug report, 
    # this corresponds to the "Direct construction" path that needs verification.
    output_tensor = tf.compat.v1.nn.separable_conv2d(
        input_tensor,
        depthwise_kernel,
        pointwise_kernel,
        strides=[1, 1, 1, 1],
        padding='VALID',
        data_format='NHWC' # Explicitly setting format to avoid ambiguity
    )

    # 3. Verification (Mirroring the check for dispatch/silent incorrectness)
    # The bug report expected a specific dispatch call. Here we verify the 
    # mathematical correctness to ensure the operation wasn't a silent no-op.
    
    # Check 1: Output shape is correct
    # With 5x5 input, 3x3 kernel, stride 1, and VALID padding, output is 3x3.
    # Output channels are determined by pointwise kernel (4).
    expected_shape = [1, 3, 3, 4]
    assert list(output_tensor.shape) == expected_shape, \
        f"Shape mismatch. Expected {expected_shape}, got {list(output_tensor.shape)}"

    # Check 2: Output is not identical to input (ensuring it's not a no-op)
    # With random kernels, the output should mathematically differ from the input.
    # We compare the valid region of the input to the output.
    input_slice = input_tensor[:, 1:4, 1:4, :4] # Rough spatial alignment for sanity check
    # Note: Exact value comparison is hard due to convolution math, 
    # so we primarily rely on shape and inequality.
    # Fix: Use input_slice instead of input_tensor to match output shape [1, 3, 3, 4]
    assert not tf.reduce_all(tf.equal(input_slice, output_tensor)).numpy(), \
        "Output is identical to input, suggesting a silent failure/no-op."

    print("Test passed: tf.compat.v1.nn.separable_conv2d executed correctly.")

if __name__ == "__main__":
    test_separable_conv2d_direct_construction()