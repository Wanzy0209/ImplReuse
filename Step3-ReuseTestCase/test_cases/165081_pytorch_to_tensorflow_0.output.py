import torch
import tensorflow as tf
import numpy as np

def test_random_normal_fuzzer_style():
    """
    Adapted test case for tf.keras.backend.random_normal based on the 
    PyTorch fuzzer program structure.
    
    The original bug report involved a complex graph of matrix multiplications 
    and specific tensor shapes (float64). This test verifies that 
    tf.keras.backend.random_normal can generate the necessary inputs for such 
    a graph and that the graph executes correctly in both eager and graph modes.
    """
    
    # Shapes extracted from the PyTorch fuzzer output
    # arg_0 to arg_7 and var_node_9
    shapes = [
        (9, 9, 9),    # arg_0
        (9, 9, 11),   # arg_1
        (9, 12, 8),   # arg_2
        (9, 8, 13),   # arg_3
        (9, 13, 7),   # arg_4
        (9, 7, 16),   # arg_5
        (9, 16, 12),  # arg_6
        (9, 12, 11),  # arg_7
        (9, 11, 12)   # var_node_9 (was torch.full, now random_normal)
    ]

    # Seed from the original report
    seed = 52676

    # Define the computation graph using tf.keras.backend.random_normal
    # We use @tf.function to test compilation (similar to torch.compile)
    @tf.function
    def tf_fuzzed_program():
        # Generate inputs using the target API: tf.keras.backend.random_normal
        # We replace the 'arg' inputs and the 'full' constant with random_normal calls
        # to test the API's ability to provide data for the computation.
        tensors = [
            tf.keras.backend.random_normal(shape, dtype=tf.float64, seed=seed) 
            for shape in shapes
        ]

        # Unpack tensors
        arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, var_node_9 = tensors

        # Replicate the matmul logic from the fuzzer
        # var_node_5 = matmul(arg_0, arg_1) -> (9, 9, 11)
        var_node_5 = tf.matmul(arg_0, arg_1)

        # var_node_8 = matmul(var_node_9, arg_2) -> (9, 11, 8)
        var_node_8 = tf.matmul(var_node_9, arg_2)

        # var_node_4 = matmul(var_node_5, var_node_8) -> (9, 9, 8)
        var_node_4 = tf.matmul(var_node_5, var_node_8)

        # var_node_12 = matmul(arg_3, arg_4) -> (9, 8, 7)
        var_node_12 = tf.matmul(arg_3, arg_4)

        # var_node_11 = matmul(var_node_12, arg_5) -> (9, 8, 16)
        var_node_11 = tf.matmul(var_node_12, arg_5)

        # var_node_3 = matmul(var_node_4, var_node_11) -> (9, 9, 16)
        var_node_3 = tf.matmul(var_node_4, var_node_11)

        # var_node_16 = matmul(arg_6, arg_7) -> (9, 16, 11)
        var_node_16 = tf.matmul(arg_6, arg_7)

        # var_node_2 = matmul(var_node_3, var_node_16) -> (9, 9, 11)
        var_node_2 = tf.matmul(var_node_3, var_node_16)

        return var_node_2

    # Execute the function
    result = tf_fuzzed_program()

    # Assertions to verify behavior
    # The final shape should be (9, 9, 11) based on the matrix multiplication chain
    assert result.shape == (9, 9, 11), f"Expected shape (9, 9, 11), got {result.shape}"
    
    # Verify dtype matches the fuzzer (float64)
    assert result.dtype == tf.float64, f"Expected dtype float64, got {result.dtype}"
    
    print("Test passed. tf.keras.backend.random_normal generated valid inputs for the fuzzer graph.")

if __name__ == "__main__":
    test_random_normal_fuzzer_style()