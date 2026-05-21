import torch
import tensorflow as tf
import numpy as np

# The original bug (Issue 164875) involves a divergence when handling tensors
# with empty dimensions (e.g., size (20, 0)) in PyTorch's torch.compile.
# This test case verifies that tf.compat.v1.reduce_all handles similar
# edge cases (empty tensors and empty dimensions) correctly.

def test_reduce_all_edge_cases():
    # 1. Test with a completely empty tensor
    # Analogous to the empty dimension issue in the bug report
    empty_tensor = tf.constant([], dtype=tf.bool)
    result = tf.compat.v1.reduce_all(empty_tensor)
    # Identity of logical AND is True
    assert result.numpy() == True, "reduce_all on empty tensor failed"

    # 2. Test with a tensor having an empty dimension (e.g., (20, 0))
    # Directly mirrors the shape (20, 0) from the PyTorch bug report
    empty_dim_tensor = tf.zeros((20, 0), dtype=tf.bool)
    result_dim = tf.compat.v1.reduce_all(empty_dim_tensor)
    assert result_dim.numpy() == True, "reduce_all on (20, 0) tensor failed"

    # 3. Test with keepdims=True on empty dimension
    result_keepdims = tf.compat.v1.reduce_all(empty_dim_tensor, keepdims=True)
    assert result_keepdims.shape == (1,), "reduce_all keepdims shape mismatch"
    assert result_keepdims.numpy()[0] == True, "reduce_all keepdims value mismatch"

    # 4. Test with specific axis reduction on empty dimension
    # Reducing the non-empty dimension (0) of a (20, 0) tensor
    result_axis = tf.compat.v1.reduce_all(empty_dim_tensor, axis=0)
    assert result_axis.shape == (0,), "reduce_all axis reduction shape mismatch"
    
    # Reducing the empty dimension (1) of a (20, 0) tensor
    result_axis_1 = tf.compat.v1.reduce_all(empty_dim_tensor, axis=1)
    assert result_axis_1.shape == (20,), "reduce_all axis reduction shape mismatch"
    assert np.all(result_axis_1.numpy() == True), "reduce_all axis reduction value mismatch"

    print(" All tests passed for tf.compat.v1.reduce_all")

if __name__ == "__main__":
    test_reduce_all_edge_cases()