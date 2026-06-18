import tensorflow as tf
import numpy as np

def test_bessel_y1_contiguity():
    """
    Test to verify that tf.compat.v1.math.special.bessel_y1 produces consistent
    results for contiguous and non-contiguous tensors, adapted from the logic
    of the PyTorch MPS linear bug report.
    """
    # Create test tensor
    # Using dimensions similar to the original bug report for context
    x = tf.random.normal((12, 64, 768))

    # Create non-contiguous tensor via transpose and reshape
    # This mimics the memory layout change (einops.rearrange) in the original PyTorch bug
    x_transposed = tf.transpose(x, [2, 0, 1])
    x_noncontig = tf.reshape(x_transposed, [768, -1])

    # Create a contiguous version by forcing a copy
    # In TensorFlow, tf.identity often creates a contiguous copy if the input is not
    x_contig = tf.identity(x_noncontig)

    print(f"Non-contiguous tensor shape: {x_noncontig.shape}")
    print(f"Contiguous tensor shape: {x_contig.shape}")

    # Apply the target API: tf.compat.v1.math.special.bessel_y1
    # This function computes the Bessel y1 function element-wise.
    result_noncontig = tf.compat.v1.math.special.bessel_y1(x_noncontig)
    result_contig = tf.compat.v1.math.special.bessel_y1(x_contig)

    # Verify consistency
    # The results should be identical regardless of memory layout
    match = np.allclose(result_noncontig.numpy(), result_contig.numpy(), atol=1e-5)
    print(f"Results match: {match}")

    if not match:
        max_diff = np.max(np.abs(result_noncontig.numpy() - result_contig.numpy()))
        print(f"Max difference: {max_diff}")
        raise AssertionError("Results differ between contiguous and non-contiguous inputs!")
    else:
        print("API handles non-contiguous tensors correctly.")

if __name__ == "__main__":
    test_bessel_y1_contiguity()