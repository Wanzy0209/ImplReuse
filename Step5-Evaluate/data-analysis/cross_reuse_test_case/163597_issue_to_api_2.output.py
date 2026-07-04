import tensorflow as tf
import numpy as np

def test_zero_padding_3d_non_contiguous():
    """
    Test case adapted from PyTorch SDPA MPS regression (Issue 163597).
    
    The original bug involved `F.scaled_dot_product_attention` producing incorrect 
    results on the MPS device when inputs were non-contiguous. This test verifies 
    that `tf.keras.layers.ZeroPadding3D` correctly handles non-contiguous (sliced) 
    tensors by comparing the output against a contiguous reference.
    """
    # 1. Setup: Create a 5D tensor (Batch, Depth, Height, Width, Channels)
    # Shape: (1, 4, 4, 4, 3)
    batch_size, depth, height, width, channels = 1, 4, 4, 4, 3
    np_data = np.arange(batch_size * depth * height * width * channels).reshape(
        batch_size, depth, height, width, channels
    ).astype(np.float32)
    
    # "Contiguous" baseline tensor
    input_contiguous = tf.constant(np_data)

    # 2. Create "Non-contiguous" tensor via slicing
    # In PyTorch, transposing or slicing creates non-contiguous tensors.
    # In TensorFlow, slicing creates a view-like operation that might have 
    # different memory strides, mimicking the condition that triggered the original bug.
    # We slice the middle 2x2x2 block.
    input_non_contiguous = input_contiguous[:, 1:3, 1:3, 1:3, :]

    # 3. Define the Layer
    padding_size = 1
    layer = tf.keras.layers.ZeroPadding3D(padding=padding_size)

    # 4. Apply Layer to Non-contiguous Input
    # This corresponds to the "fast path" behavior in the original bug report
    output_non_contiguous = layer(input_non_contiguous)

    # 5. Apply Layer to Contiguous Input and Slice Result
    # This corresponds to the "CPU" or "Manual" reference behavior in the original bug report
    output_contiguous_full = layer(input_contiguous)
    
    # Calculate the expected slice indices.
    # The non-contiguous input is a slice [1:3] of the original input.
    # In the fully padded tensor (output_contiguous_full), this data is located at 
    # indices [1+padding : 3+padding].
    # Since output_non_contiguous has padding added around the slice [1:3],
    # we need to extract the region from output_contiguous_full that includes 
    # this data PLUS the padding around it.
    # Therefore, we extend the range by 'padding_size' on both sides.
    # Start index: (1 + padding_size) - padding_size = 1
    # End index: (3 + padding_size) + padding_size = 3 + 2 * padding_size
    start_idx = 1
    end_idx = 3 + 2 * padding_size
    
    output_expected = output_contiguous_full[:, start_idx:end_idx, start_idx:end_idx, start_idx:end_idx, :]

    # 6. Compare Results
    # The original bug showed a large norm difference between the buggy path and the reference.
    # Here we check for exact or near-exact equality.
    diff = tf.reduce_max(tf.abs(output_non_contiguous - output_expected))
    
    print(f"Max difference between non-contiguous and expected output: {diff.numpy()}")

    # Assert that the difference is negligible (floating point tolerance)
    # If the layer fails to handle non-contiguous inputs correctly, this will fail.
    assert diff < 1e-5, (
        f"ZeroPadding3D regression detected: Results differ for non-contiguous inputs. "
        f"Max diff: {diff.numpy()}"
    )
    
    print("Test passed: ZeroPadding3D handles non-contiguous tensors correctly.")

if __name__ == "__main__":
    test_zero_padding_3d_non_contiguous()