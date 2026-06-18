import torch
import tensorflow as tf
import numpy as np

def test_lecun_normal_redundant_dtype_conversion():
    """
    Test case reflecting the redundant dtype conversion logic found in the bug report.
    
    The original bug involved creating a tensor with a specific dtype (torch.zeros),
    performing an in-place operation, and then redundantly converting it to the same dtype.
    
    This test applies the same logic to tf.keras.initializers.LecunNormal:
    1. Initialize a tensor with a specific dtype.
    2. Perform an operation (masking).
    3. Attempt a redundant cast to the same dtype.
    4. Verify that the cast was unnecessary (dtype was already correct).
    """
    # Setup parameters
    shape = (10, 10)
    target_dtype = tf.float32
    
    # 1. Initialize tensor with specific dtype (Analogous to torch.zeros(..., dtype=query.dtype))
    initializer = tf.keras.initializers.LecunNormal()
    weights = initializer(shape, dtype=target_dtype)
    
    # Verify initialization respected the dtype
    assert weights.dtype == target_dtype, \
        f"LecunNormal failed to initialize with dtype {target_dtype}, got {weights.dtype}"

    # 2. Perform an operation (Analogous to masked_fill_)
    # We create a mask and apply it. Note: TF operations are not in-place, 
    # so we assign to a new variable, but the dtype should persist.
    mask = tf.ones(shape, dtype=tf.bool)
    operated_weights = tf.where(mask, weights, tf.zeros_like(weights))
    
    # Verify operation preserved the dtype
    assert operated_weights.dtype == target_dtype, \
        f"Operation changed dtype to {operated_weights.dtype}, expected {target_dtype}"

    # 3. Perform redundant conversion (Analogous to attn_bias.to(query.dtype))
    # In TensorFlow, .to() is equivalent to tf.cast()
    redundant_weights = tf.cast(operated_weights, target_dtype)

    # 4. Verify the redundancy
    # The redundant cast should result in the same dtype and values
    assert redundant_weights.dtype == target_dtype
    assert np.allclose(operated_weights.numpy(), redundant_weights.numpy()), \
        "Redundant cast altered tensor values unexpectedly"

    print("Test passed: LecunNormal initialization respects dtype, rendering subsequent cast redundant.")

if __name__ == "__main__":
    test_lecun_normal_redundant_dtype_conversion()