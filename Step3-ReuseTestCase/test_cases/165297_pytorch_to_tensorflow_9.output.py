import torch
import tensorflow as tf
import numpy as np

def test_sparse_add_bfloat16_large():
    """
    Adapted test case for tf.compat.v1.sparse_add based on the PyTorch MaxPool2d bug.
    
    Original Bug Context:
    - API: torch.nn.MaxPool2d
    - Issue: Large tensors with bfloat16 and channels_last memory format produced NaNs.
    
    Adaptation Logic:
    - API: tf.compat.v1.sparse_add
    - Translation: Test the robustness of sparse_add with large tensor dimensions 
      and bfloat16 dtype to check for similar numerical instability (NaNs/Infs).
    """
    # Dimensions similar to the PyTorch bug report (Large tensor)
    N, C, H, W = 84, 64, 512, 960
    dense_shape = [N, C, H, W]

    # Create large sparse tensors to stress the operation
    # Using a significant number of non-zero elements to simulate load
    num_non_zero = 50000

    # Generate random indices
    # SparseTensor indices must be sorted in lexicographic order
    indices = np.random.randint(0, [N, C, H, W], size=(num_non_zero, 4), dtype=np.int64)
    indices.sort(axis=0)

    # Generate random values
    # PyTorch bug specifically involved bfloat16
    values_a = np.random.randn(num_non_zero).astype(np.float32)
    values_b = np.random.randn(num_non_zero).astype(np.float32)

    # Create SparseTensors with bfloat16 dtype
    sp_a = tf.SparseTensor(
        indices=indices,
        values=tf.cast(values_a, tf.bfloat16),
        dense_shape=dense_shape
    )

    sp_b = tf.SparseTensor(
        indices=indices,
        values=tf.cast(values_b, tf.bfloat16),
        dense_shape=dense_shape
    )

    print(f"Input shape: {dense_shape}")
    print(f"Input dtype: {sp_a.values.dtype}")
    print(f"Number of non-zero elements: {num_non_zero}")

    # Perform the sparse_add operation
    # This is the equivalent of the "operation under test"
    result = tf.compat.v1.sparse_add(sp_a, sp_b)

    # Check for NaNs and Infs in the result values
    # The original bug reported NaNs in the output
    has_nan = tf.reduce_any(tf.math.is_nan(result.values))
    has_inf = tf.reduce_any(tf.math.is_inf(result.values))

    print(f"Output contains NaN? {has_nan.numpy()}")
    print(f"Output contains Inf? {has_inf.numpy()}")

    # Assertions to verify behavior
    assert not has_nan.numpy(), "Detected NaNs in sparse_add output!"
    assert not has_inf.numpy(), "Detected Infs in sparse_add output!"

if __name__ == "__main__":
    test_sparse_add_bfloat16_large()