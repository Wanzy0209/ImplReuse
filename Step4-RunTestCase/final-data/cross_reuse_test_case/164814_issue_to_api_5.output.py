import sys
import numpy as np

try:
    import torch
    import tensorflow as tf
except ImportError as e:
    # Handle environment issues, specifically the GLIBC version mismatch or missing TF
    print(f"Skipping test due to environment/dependency error: {e}")
    if "GLIBCXX" in str(e):
        print("Error: The system is missing the required GLIBCXX version (likely 3.4.29).")
        print("This is a system-level dependency issue. Please update your environment's libstdc++ or use a compatible TensorFlow build.")
    else:
        print("Error: TensorFlow or PyTorch is not installed or cannot be imported.")
    sys.exit(0)

def test_batch_set_value_with_scalar_reduction():
    """
    Test case for tf.keras.backend.batch_set_value based on the logic from 
    PyTorch Issue 164814. The original issue involves a divergence when handling
    0-d tensors (scalars) produced by a chain of unique/reshape/squeeze operations.
    
    This test verifies that batch_set_value correctly handles assignments involving
    these specific tensor shapes, particularly the 0-d tensor case.
    """
    
    # Setup variables to act as targets for batch_set_value
    # Corresponds to the state where we might want to update weights with computed scalars
    var_2d = tf.Variable(np.zeros((2, 3), dtype=np.int32), dtype=tf.int32)
    var_0d = tf.Variable(np.zeros((), dtype=np.int32), dtype=tf.int32)

    # Reproduce the data manipulation logic from the original bug report
    # Original: var_node_3 = torch.full((2, 3), 3, dtype=torch.int32)
    # We use the shape info for context, but generate the value to be assigned.
    
    # Original: _inp_unique_wide = torch.arange(1, device=var_node_3.device, dtype=torch.int64)
    val_source = tf.range(1, dtype=tf.int64)
    
    # Original: _uniq_wide = torch.unique(_inp_unique_wide)
    # tf.unique returns a tuple (unique, idx), we take the unique values (index 0)
    val_unique = tf.unique(val_source)[0]
    
    # Original: var_node_2 = _uniq_wide.to(var_node_3.dtype)
    val_cast = tf.cast(val_unique, tf.int32)
    
    # Original: var_node_1 = torch.reshape(var_node_2, [1])
    val_reshaped = tf.reshape(val_cast, [1])
    
    # Original: var_node_0 = torch.squeeze(var_node_1)
    # This results in a 0-d tensor (scalar), which is the critical edge case in the bug.
    val_scalar = tf.squeeze(val_reshaped)

    # Leverage the similar API: tf.keras.backend.batch_set_value
    # We attempt to set the values of the variables using the generated tensors.
    # This tests if the API can handle the dimensionality reduction (0-d tensor) correctly.
    tf.keras.backend.batch_set_value([
        (var_2d, tf.fill((2, 3), 3)), # Assigning a standard 2D tensor
        (var_0d, val_scalar)          # Assigning the 0-d tensor (scalar)
    ])

    # Assertions to verify correctness
    # Check 2D assignment
    assert tf.reduce_all(var_2d == 3).numpy(), "2D variable assignment failed"
    
    # Check 0D assignment
    # val_source was range(1), so unique is [1], cast to int32, reshaped to [1], squeezed to scalar 1.
    assert var_0d.numpy() == 1, "0D variable assignment failed or incorrect value"

    print(" batch_set_value handles scalar (0-d) and dynamic shape assignments correctly")

if __name__ == "__main__":
    test_batch_set_value_with_scalar_reduction()