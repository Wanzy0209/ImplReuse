import tensorflow as tf

def test_bessel_y1_contiguity():
    """
    Adapted test case for tf.math.special.bessel_y1 based on PyTorch Issue 162730.
    Verifies that the API produces consistent results for non-contiguous vs contiguous tensors.
    """
    # Create test tensors
    # Mimicking the shape from the PyTorch bug report (12, 64, 768)
    x = tf.random.normal((12, 64, 768))

    # Create non-contiguous tensor via transpose + reshape (mimicking einops.rearrange)
    # This sequence of operations often results in a tensor with non-standard memory strides
    x_noncontig = tf.reshape(tf.transpose(x, perm=[2, 0, 1]), [768, 768])

    # Create a contiguous copy
    # In TensorFlow, tf.identity is typically used to break the view graph and ensure a copy
    x_contig = tf.identity(x_noncontig)

    # Run the API on both versions
    result_noncontig = tf.math.special.bessel_y1(x_noncontig)
    result_contig = tf.math.special.bessel_y1(x_contig)

    # Verify results match
    # Using a tolerance similar to the original PyTorch test
    match = tf.reduce_all(tf.abs(result_noncontig - result_contig) < 1e-5)
    max_diff = tf.reduce_max(tf.abs(result_noncontig - result_contig))

    print(f"Results match: {match.numpy()}")
    print(f"Max difference: {max_diff.numpy()}")

    # Assert to ensure correctness
    assert match.numpy(), f"Results differ between non-contiguous and contiguous inputs! Max diff: {max_diff.numpy()}"

if __name__ == "__main__":
    test_bessel_y1_contiguity()