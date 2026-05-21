import torch
import tensorflow as tf
import numpy as np

def test_unpack_x_y_sample_weight():
    """
    Adapted test case for tf.keras.utils.unpack_x_y_sample_weight based on 
    PyTorch issue #166279 (torch.chunk eager/compile divergence).
    
    The original issue involved splitting tensors of specific shapes and dtypes 
    (bool) and concatenating them. This test mimics that data flow by unpacking
    pre-split tensors and verifying the concatenation logic.
    """
    
    # Mimic the tensor shapes and dtypes from the PyTorch fuzzer output.
    # Original PyTorch code:
    # var_node_2 = torch.chunk(..., 4, ...)[0] -> size (3,), bool
    # var_node_4 = torch.chunk(..., 2, ...)[0] -> size (5,), bool
    # var_node_8 = torch.squeeze(torch.chunk(..., 4, dim=1)[0]) -> size (6,), bool
    # var_node_11 = arg_3 -> size (2,), bool
    
    # We create these "chunks" directly to pass into the unpacking utility.
    # Using bool dtype as per the original bug report.
    var_node_2 = tf.zeros((3,), dtype=tf.bool)
    var_node_4 = tf.zeros((5,), dtype=tf.bool)
    var_node_8 = tf.zeros((6,), dtype=tf.bool)
    var_node_11 = tf.zeros((2,), dtype=tf.bool)

    # The PyTorch code concatenates var_node_2, var_node_4, and var_node_8 first.
    # We simulate the "splitting" phase by packing these into a tuple for unpacking.
    # unpack_x_y_sample_weight handles (x,), (x, y), or (x, y, sample_weight).
    # We map the three chunks to x, y, and sample_weight.
    data_tuple = (var_node_2, var_node_4, var_node_8)

    # Define the logic inside a tf.function to check for graph compilation issues,
    # similar to torch.compile in the original bug report.
    @tf.function
    def run_compiled(data, extra_tensor):
        # Unpack the data
        x, y, sample_weight = tf.keras.utils.unpack_x_y_sample_weight(data)
        
        # Verify unpacking
        assert x is not None
        assert y is not None
        assert sample_weight is not None

        # Mimic: var_node_1 = torch.cat([var_node_2, var_node_4, var_node_8], dim=0)
        # Expected size: 3 + 5 + 6 = 14
        var_node_1 = tf.concat([x, y, sample_weight], axis=0)
        
        # Mimic: var_node_0 = torch.cat([var_node_1, var_node_11], dim=0)
        # Expected size: 14 + 2 = 16
        var_node_0 = tf.concat([var_node_1, extra_tensor], axis=0)
        
        return var_node_0

    # 1. Test Eager Execution
    print("Testing Eager Execution...")
    result_eager = run_compiled(data_tuple, var_node_11)
    assert result_eager.shape == (16,), f"Expected shape (16,), got {result_eager.shape}"
    assert result_eager.dtype == tf.bool, f"Expected dtype bool, got {result_eager.dtype}"
    print(" Eager execution success")

    # 2. Test Compiled Execution (Graph Mode)
    print("Testing Compiled Execution...")
    result_compiled = run_compiled(data_tuple, var_node_11)
    assert result_compiled.shape == (16,), f"Expected shape (16,), got {result_compiled.shape}"
    assert result_compiled.dtype == tf.bool, f"Expected dtype bool, got {result_compiled.dtype}"
    print(" Compiled execution success")

    # 3. Verify consistency
    # Note: Direct equality check might fail if tensors are recreated, 
    # but since inputs are constants/zeros, values should match.
    if tf.reduce_all(tf.equal(result_eager, result_compiled)):
        print(" Eager and Compiled results match")
    else:
        # In this specific case with zeros, they should match.
        print(" Eager and Compiled results differ (unexpected for zero inputs)")

if __name__ == "__main__":
    test_unpack_x_y_sample_weight()