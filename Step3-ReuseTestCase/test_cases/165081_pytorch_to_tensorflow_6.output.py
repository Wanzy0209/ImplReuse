import torch
import tensorflow as tf
import numpy as np

def test_convert_to_tensor_fuzzer_divergence():
    """
    Adapts the PyTorch fuzzer test case to TensorFlow.
    The original issue involves eager/compile divergence in PyTorch Dynamo.
    This test verifies the stability of tf.keras.ops.convert_to_tensor
    and tf.matmul operations within a tf.function (compiled) context,
    using float64 dtypes and specific tensor shapes from the fuzzer.
    """
    
    # Setup seed to mimic the original environment
    np.random.seed(52676)
    tf.random.set_seed(52676)

    # Define input shapes based on the comments in the PyTorch bug report
    # We use numpy arrays as inputs to test the conversion capability
    input_shapes = [
        (9, 9, 9),   # arg_0
        (9, 9, 11),  # arg_1
        (9, 12, 8),  # arg_2
        (9, 8, 13),  # arg_3
        (9, 13, 7),  # arg_4
        (9, 7, 16),  # arg_5
        (9, 16, 12), # arg_6
        (9, 12, 11), # arg_7
        (9, 11, 8),  # arg_8
        (9, 8, 7),   # arg_9
        (9, 7, 16),  # arg_10
        (9, 16, 11), # arg_11
        (9, 9, 11),  # arg_12
        (156, 9),    # arg_13
        (9, 13),     # arg_14
        (9, 11, 12), # arg_15
        (9, 12, 8),  # arg_16
        (9, 8, 13),  # arg_17
        (9, 13, 7)   # arg_18
    ]

    # Create dummy numpy inputs
    args = [np.random.randn(*s).astype(np.float64) for s in input_shapes]
    sentinel = None

    # The target API: tf.keras.ops.convert_to_tensor
    # We wrap the logic in tf.function to mimic the compilation aspect of torch._dynamo
    @tf.function
    def fuzzed_program_tf(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 
                          arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, 
                          arg_15, arg_16, arg_17, arg_18, sentinel):
        
        dtype = tf.float64

        # Convert inputs using the target API
        var_node_6 = tf.keras.ops.convert_to_tensor(arg_0, dtype=dtype)
        var_node_7 = tf.keras.ops.convert_to_tensor(arg_1, dtype=dtype)
        
        # Replicate torch.full using convert_to_tensor on numpy arrays
        var_node_9 = tf.keras.ops.convert_to_tensor(
            np.full((9, 11, 12), 1.5758497316910556), dtype=dtype
        )
        
        var_node_10 = tf.keras.ops.convert_to_tensor(arg_2, dtype=dtype)
        
        var_node_13 = tf.keras.ops.convert_to_tensor(arg_3, dtype=dtype)
        var_node_14 = tf.keras.ops.convert_to_tensor(arg_4, dtype=dtype)
        
        var_node_15 = tf.keras.ops.convert_to_tensor(arg_5, dtype=dtype)
        
        var_node_17 = tf.keras.ops.convert_to_tensor(arg_6, dtype=dtype)
        var_node_18 = tf.keras.ops.convert_to_tensor(arg_7, dtype=dtype)

        # Additional constants from the snippet
        var_node_23 = tf.keras.ops.convert_to_tensor(
            np.full((156, 8), -0.5249394453404403), dtype=dtype
        )
        var_node_24 = tf.keras.ops.convert_to_tensor(
            np.full((8, 9), 0.9331226188585692), dtype=dtype
        )
        var_node_26 = tf.keras.ops.convert_to_tensor(
            np.full((9, 13), -0.9276381954691514), dtype=dtype
        )

        # Replicate the matmul chain
        var_node_5 = tf.matmul(var_node_6, var_node_7)
        var_node_8 = tf.matmul(var_node_9, var_node_10)
        var_node_4 = tf.matmul(var_node_5, var_node_8)
        
        var_node_12 = tf.matmul(var_node_13, var_node_14)
        var_node_11 = tf.matmul(var_node_12, var_node_15)
        var_node_3 = tf.matmul(var_node_4, var_node_11)
        
        var_node_16 = tf.matmul(var_node_17, var_node_18)
        var_node_2 = tf.matmul(var_node_3, var_node_16)
        
        var_node_22 = tf.matmul(var_node_23, var_node_24)
        
        # Return a result to verify execution flow
        return var_node_2

    # Execute the test
    try:
        result = fuzzed_program_tf(*args, sentinel)
        # Verify shape matches the PyTorch output comment for var_node_2
        assert result.shape == (9, 9, 11), f"Expected shape (9, 9, 11), got {result.shape}"
        print("Test passed: tf.keras.ops.convert_to_tensor handled the fuzzer inputs correctly.")
    except Exception as e:
        print(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    test_convert_to_tensor_fuzzer_divergence()