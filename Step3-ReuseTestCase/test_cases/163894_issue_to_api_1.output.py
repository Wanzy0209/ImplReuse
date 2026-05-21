import torch
import tensorflow as tf
import numpy as np

def test_tf_nonzero_divergence():
    """
    Test case adapted from PyTorch Issue 163894.
    
    The original issue highlights a divergence between eager and compiled modes
    regarding stride hints and dynamic shapes (specifically with torch.nonzero).
    
    This test leverages the similar API 'tf.compat.v1.experimental.output_all_intermediates'
    to check the execution environment and then verifies that TensorFlow handles
    the equivalent dynamic shape operations (tf.where) consistently between
    eager and graph modes.
    """
    
    # Leverage the similar API to check the configuration for intermediates.
    # This relates to the bug's theme of how execution modes (eager vs graph)
    # handle internal state and outputs.
    output_intermediates = tf.compat.v1.experimental.output_all_intermediates()
    print(f"Output all intermediates enabled: {output_intermediates}")

    # Define the logic mirroring the PyTorch fuzzed_program
    def fuzzed_program(arg_0):
        # var_node_1 = arg_0 # size=(1, 2), dtype=int64
        var_node_1 = arg_0

        # var_node_5 = torch.full((1, 2), -66, dtype=torch.int32)
        var_node_5 = tf.fill([1, 2], tf.constant(-66, dtype=tf.int32))

        # var_node_6 = torch.full((1, 2), 77, dtype=torch.int64)
        var_node_6 = tf.fill([1, 2], tf.constant(77, dtype=tf.int64))

        # var_node_4 = add(var_node_5, var_node_6) -> int32 in PyTorch
        # Cast to int32 to match PyTorch's type promotion rules (int32 + int64 -> int32)
        var_node_4 = tf.add(var_node_5, tf.cast(var_node_6, tf.int32))

        # var_node_7 = torch.full((1, 2), -64, dtype=torch.int32)
        var_node_7 = tf.fill([1, 2], tf.constant(-64, dtype=tf.int32))

        # var_node_3 = mul(var_node_4, var_node_7)
        var_node_3 = tf.multiply(var_node_4, var_node_7)

        # var_node_9 = torch.full((3, 4), False, dtype=torch.bool)
        var_node_9 = tf.fill([3, 4], False)

        # var_node_8 = torch.nonzero(var_node_9)
        # In TF, tf.where with 1 argument returns indices of True elements.
        # Since all are False, this returns a tensor of shape (0, 2).
        # This dynamic shape is the source of the divergence in the original bug.
        var_node_8 = tf.where(var_node_9)

        # var_node_2 = add(var_node_3, var_node_8)
        # Broadcasting (1, 2) with (0, 2) -> (0, 2)
        var_node_2 = tf.add(var_node_3, var_node_8)

        # var_node_0 = div(var_node_1, var_node_2)
        # Broadcasting (1, 2) with (0, 2) -> (0, 2)
        # Cast to float32 for division to match standard behavior
        var_node_0 = tf.math.divide(tf.cast(var_node_1, tf.float32), tf.cast(var_node_2, tf.float32))

        return var_node_0

    # Setup inputs
    # arg_0: size=(1, 2), dtype=int64
    arg_0 = tf.constant([[1, 2]], dtype=tf.int64)

    # 1. Run in Eager Mode
    print("Running in Eager Mode...")
    result_eager = fuzzed_program(arg_0)
    print(f"Eager Result Shape: {result_eager.shape}")
    print(f"Eager Result: {result_eager.numpy()}")

    # 2. Run in Graph Mode (Compiled)
    print("\nRunning in Graph Mode (tf.function)...")
    compiled_program = tf.function(fuzzed_program)
    result_graph = compiled_program(arg_0)
    print(f"Graph Result Shape: {result_graph.shape}")
    print(f"Graph Result: {result_graph.numpy()}")

    # 3. Assert Consistency
    # The original bug was a divergence between eager and compiled modes.
    # We assert that the shapes and values match.
    assert result_eager.shape == result_graph.shape, \
        f"Shape mismatch: Eager {result_eager.shape} vs Graph {result_graph.shape}"
    
    # Use numpy for value comparison
    np.testing.assert_array_equal(result_eager.numpy(), result_graph.numpy())
    
    print("\n Test Passed: Eager and Graph modes are consistent.")

if __name__ == "__main__":
    test_tf_nonzero_divergence()