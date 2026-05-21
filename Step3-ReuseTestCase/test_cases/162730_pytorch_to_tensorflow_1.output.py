import tensorflow as tf
import numpy as np

def test_bessel_y0_contiguous_consistency():
    """
    Adapted test case for tf.math.special.bessel_y0 based on the 
    PyTorch MPS F.Linear bug (Issue 162730).
    
    Verifies that the API produces consistent results between 
    non-contiguous (transposed) and contiguous (copied) tensors.
    """
    
    # Create test tensors
    # Using a shape similar to the original bug report
    x = tf.random.normal((12, 64, 768))
    
    # Create non-contiguous tensor via transpose
    # In PyTorch, rearranging/transpose creates a view with modified strides.
    # In TensorFlow, transpose creates a tensor with non-standard strides.
    x_noncontig = tf.transpose(x, perm=[2, 0, 1])
    
    # Create contiguous tensor
    # In PyTorch, .contiguous() forces a copy if the tensor is not contiguous.
    # In TensorFlow, we use tf.identity to force a copy/ensure standard layout.
    x_contig = tf.identity(x_noncontig)
    
    # Run the similar API
    result_noncontig = tf.math.special.bessel_y0(x_noncontig)
    result_contig = tf.math.special.bessel_y0(x_contig)
    
    # These should be identical
    # The original bug showed a mismatch on MPS for PyTorch.
    # We assert they match for TensorFlow.
    try:
        np.testing.assert_allclose(result_noncontig.numpy(), result_contig.numpy(), atol=1e-5)
        print("Results match: True")
        print("Max difference:", np.abs(result_noncontig.numpy() - result_contig.numpy()).max())
    except AssertionError as e:
        print("Results match: False")
        print(f"Assertion Error: {e}")
        raise

if __name__ == "__main__":
    test_bessel_y0_contiguous_consistency()