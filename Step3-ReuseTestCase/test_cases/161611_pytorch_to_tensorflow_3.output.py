import torch
import tensorflow as tf
import numpy as np

def test_geomspace_redundant_dtype_conversion():
    """
    Adapted from PyTorch Issue 161611.
    
    This test verifies that tf.experimental.numpy.geomspace correctly handles the 
    dtype argument, ensuring that a subsequent redundant conversion (similar to 
    the .to() call in the original bug report) is unnecessary and ineffective 
    if not assigned.
    """
    
    # Setup parameters mimicking the context of the original bug
    # In the original bug, 'query.dtype' was the target.
    target_dtype = tf.float32
    start = 1.0
    stop = 10.0
    num = 5

    # 1. Create the tensor using the API with the specific dtype
    # Equivalent to: attn_bias = torch.zeros(..., dtype=query.dtype)
    data = tf.experimental.numpy.geomspace(start, stop, num=num, dtype=target_dtype)

    # Verify that the API respects the dtype argument immediately
    assert data.dtype == target_dtype, \
        f"Expected dtype {target_dtype}, but got {data.dtype}"

    # 2. Perform an operation (mimicking the masked_fill_ logic in the original bug)
    # We use a simple multiplication to simulate an operation that preserves dtype.
    # Note: TF tensors are immutable, so this creates a new tensor.
    data_processed = data * 1.0 

    # 3. Perform the redundant conversion
    # Equivalent to: attn_bias.to(query.dtype)
    # In the original bug, this line was redundant because 'attn_bias' was already 
    # 'query.dtype', and the result was not assigned.
    # Here, we call .astype() (TF/NumPy equivalent of .to()) without assignment.
    data_processed.astype(target_dtype)

    # 4. Verify that the unassigned conversion did not alter the variable
    # The variable should still be the target_dtype.
    assert data_processed.dtype == target_dtype, \
        "Dtype changed unexpectedly after redundant unassigned conversion"

    # 5. Verify idempotency: explicitly converting to the same dtype should not change values
    data_redundant = data_processed.astype(target_dtype)
    assert data_redundant.dtype == target_dtype
    assert np.array_equal(data_processed.numpy(), data_redundant.numpy()), \
        "Values changed after redundant dtype conversion"

    print("Test passed: Redundant dtype conversion behaves as expected.")

if __name__ == "__main__":
    test_geomspace_redundant_dtype_conversion()