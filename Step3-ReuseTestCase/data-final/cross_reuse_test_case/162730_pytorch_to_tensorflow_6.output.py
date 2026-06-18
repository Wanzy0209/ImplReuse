import tensorflow as tf
import numpy as np

def test_bessel_j1_contiguity():
    """
    Adapted test case for tf.compat.v1.math.special.bessel_j1 based on 
    PyTorch MPS F.Linear contiguity bug (Issue 162730).
    
    Verifies that the API produces consistent results for contiguous 
    and non-contiguous input tensors.
    """
    
    # Create test tensors
    # Using shapes similar to the original bug report context
    W = tf.random.normal((12, 64, 768))

    # Create non-contiguous tensor via transpose + reshape
    # This mimics the einops.rearrange(W, "h d m -> m (h d)") logic
    # which changes the memory access pattern (strides)
    x_noncontig = tf.reshape(tf.transpose(W, [2, 0, 1]), [768, -1])

    # Create contiguous version
    # In TensorFlow, we force a contiguous copy by converting to numpy and back
    # to ensure the memory layout is standard/contiguous.
    x_contig = tf.convert_to_tensor(x_noncontig.numpy())

    # Run the API under test
    result1 = tf.compat.v1.math.special.bessel_j1(x_noncontig)
    result2 = tf.compat.v1.math.special.bessel_j1(x_contig)

    # Verify results
    # Since bessel_j1 is element-wise, results should be identical regardless of memory layout
    match = np.allclose(result1.numpy(), result2.numpy(), atol=1e-5)
    max_diff = np.abs(result1.numpy() - result2.numpy()).max()

    print(f"Results match: {match}")
    print(f"Max difference: {max_diff}")

    # Assertion to act as a regression test
    assert match, f"Results differ between contiguous and non-contiguous inputs. Max diff: {max_diff}"

if __name__ == "__main__":
    test_bessel_j1_contiguity()