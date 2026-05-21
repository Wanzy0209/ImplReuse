import torch
import tensorflow as tf
import numpy as np

def test_broadcast_dynamic_shape_compilation():
    """
    Test case adapted from PyTorch Issue 166153.
    
    Original Issue Context:
    Models using flex_attention with ao hit the recompile_limit because 
    dynamic input validation (checking object IDs/dtypes) triggered excessive 
    recompilations in torch.compile.

    Adaptation Logic:
    This test verifies that tf.broadcast_dynamic_shape handles dynamic shape 
    inputs within a compiled context (tf.function) robustly. It checks that 
    the API correctly computes broadcast shapes for various dynamic inputs 
    and properly validates incompatible shapes, mirroring the validation 
    logic that caused issues in the original PyTorch bug.
    """

    # Define a compiled function (analogous to torch.compile in the issue)
    @tf.function
    def compute_broadcast(shape_x, shape_y):
        """
        Computes the shape of a broadcast given symbolic shapes.
        This involves internal validation logic similar to the 
        _validate_sdpa_input check mentioned in the PyTorch logs.
        """
        return tf.broadcast_dynamic_shape(shape_x, shape_y)

    # Scenario 1: Valid dynamic shapes
    # Simulating the loop that caused the recompile limit in PyTorch
    # by passing multiple different shape tensors.
    shape_pairs = [
        (tf.constant([1, 2, 3]), tf.constant([5, 1, 3])),
        (tf.constant([2, 1]), tf.constant([1, 3])),
        (tf.constant([1, 4, 5]), tf.constant([3, 1, 1])),
    ]

    expected_results = [
        tf.constant([5, 2, 3]),
        tf.constant([2, 3]),
        tf.constant([3, 4, 5]),
    ]

    for i, (sx, sy) in enumerate(shape_pairs):
        result = compute_broadcast(sx, sy)
        # Assert correctness to ensure the compiled graph works for these inputs
        assert tf.reduce_all(result == expected_results[i]).numpy(), \
            f"Failed for shapes {sx.numpy()} and {sy.numpy()}"

    # Scenario 2: Incompatible shapes (Validation check)
    # Mirrors the validation failure in the PyTorch log:
    # "if query.dtype != key.dtype ... _validate_sdpa_input"
    # Here we check if the API correctly raises an error for incompatible shapes.
    invalid_x = tf.constant([2, 3])
    invalid_y = tf.constant([4, 5])

    # We expect the compiled function to handle the validation error correctly
    # without crashing the compilation process itself.
    with tf.assertRaises(tf.errors.InvalidArgumentError):
        compute_broadcast(invalid_x, invalid_y)

    print("Test passed: broadcast_dynamic_shape handles dynamic inputs and validation correctly.")

if __name__ == "__main__":
    test_broadcast_dynamic_shape_compilation()