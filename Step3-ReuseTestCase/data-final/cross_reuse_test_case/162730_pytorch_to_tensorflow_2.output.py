import torch
import tensorflow as tf
import numpy as np

def test_bessel_j0_noncontiguous():
    """
    Adapted test case for tf.compat.v1.math.special.bessel_j0.
    Verifies that the API produces consistent results for contiguous 
    and non-contiguous input tensors, mirroring the logic of the 
    PyTorch MPS linear bug.
    """
    
    # Create test tensors
    # Original PyTorch: W = torch.randn(12, 64, 768, device=device)
    # Using float32 as bessel_j0 supports float/half/float64
    W = tf.random.normal((12, 64, 768), dtype=tf.float32)

    # Create non-contiguous tensor via transpose and reshape
    # PyTorch: w_noncontig = einops.rearrange(W, "h d m -> m (h d)")
    # TensorFlow equivalent: Transpose dimensions to change memory layout, then reshape
    w_transposed = tf.transpose(W, perm=[2, 1, 0])
    w_noncontig = tf.reshape(w_transposed, (768, 12 * 64))

    # Create a contiguous version
    # In TensorFlow, converting to numpy and back forces a contiguous memory copy
    w_contig = tf.Variable(initial_value=w_noncontig.numpy())

    # Apply the API: tf.compat.v1.math.special.bessel_j0
    # Original: torch.nn.functional.linear(x, w_noncontig, bias)
    # Here we pass the manipulated tensors as the input 'x'
    result_noncontig = tf.compat.v1.math.special.bessel_j0(w_noncontig)
    result_contig = tf.compat.v1.math.special.bessel_j0(w_contig)

    # Check results
    # Original: torch.allclose(result1, result2, atol=1e-5)
    match = np.allclose(result_noncontig.numpy(), result_contig.numpy(), atol=1e-5)
    max_diff = np.max(np.abs(result_noncontig.numpy() - result_contig.numpy()))

    print(f"Results match: {match}")
    print(f"Max difference: {max_diff}")

    # Assertion to catch potential bugs
    assert match, f"Results differ by {max_diff}"

if __name__ == "__main__":
    test_bessel_j0_noncontiguous()