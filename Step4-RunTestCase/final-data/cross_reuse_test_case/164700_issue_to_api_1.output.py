import sys

try:
    import torch
    import tensorflow as tf
    import numpy as np
except ImportError as e:
    print(f"Skipping test due to import error: {e}")
    print("This is likely due to a missing or incompatible system library (e.g., libstdc++ GLIBCXX_3.4.29).")
    sys.exit(0)

def test_mirrored_tensor_interleaving():
    """
    Test case adapted from PyTorch Issue 164700.
    Verifies that the interleaving of unsqueeze (expand_dims), concatenation,
    and reshaping operations works correctly within the TensorFlow 
    MirroredStrategy context (tf.types.experimental.distributed.Mirrored).
    """
    
    # Initialize the MirroredStrategy to utilize the Mirrored API
    strategy = tf.distribute.MirroredStrategy()

    print(f"Number of devices: {strategy.num_replicas_in_sync}")

    # Define the logic adapted from the PyTorch bug report
    # Original PyTorch operations: unsqueeze (None indexing), torch.cat, reshape, torch.arange
    def logic_fn(x, y):
        # Cast x to int32 to match y's type for concatenation (TF is stricter than PyTorch here)
        x = tf.cast(x, tf.int32)

        # y2 = torch.cat([x[:, 1:], y[:, None] + 32 * 2048], dim=1)
        # TF: tf.concat, tf.expand_dims
        y_expanded = tf.expand_dims(y, axis=1)
        y_offset = y_expanded + 32 * 2048
        x_slice = x[:, 1:]
        y2 = tf.concat([x_slice, y_offset], axis=1)

        # x2 = x[:, 1:, None]
        x2 = tf.expand_dims(x[:, 1:], axis=1)

        # y3 = y2[:, -1:, None]
        y3 = tf.expand_dims(y2[:, -1:], axis=1)

        # return (torch.cat([x2, y3], dim=1) + torch.arange(-2048, 0, device=device)[None, None, :]).reshape(1, 32 * 2048)
        concat_res = tf.concat([x2, y3], axis=1)
        
        # Create the arange tensor and broadcast it
        # PyTorch: torch.arange(-2048, 0)[None, None, :]
        arange_tensor = tf.range(-2048, 0, dtype=tf.int32)
        arange_expanded = tf.reshape(arange_tensor, (1, 1, -1))
        
        added = concat_res + arange_expanded
        return tf.reshape(added, (1, 32 * 2048))

    # Input creation functions for distributing values to Mirrored devices
    def x_input_fn(ctx):
        batch_size = ctx.get_per_replica_batch_size(1)
        return tf.zeros((batch_size, 32), dtype=tf.int32)

    def y_input_fn(ctx):
        batch_size = ctx.get_per_replica_batch_size(1)
        return tf.zeros((batch_size,), dtype=tf.int32)

    # Create Distributed Values (Mirrored)
    dist_x = strategy.experimental_distribute_values_from_function(x_input_fn)
    dist_y = strategy.experimental_distribute_values_from_function(y_input_fn)

    # Run the computation step
    @tf.function
    def run_step(x, y):
        return logic_fn(x, y)

    print("Running distributed computation...")
    result = strategy.run(run_step, args=(dist_x, dist_y))

    # Extract results from the PerReplica object
    # In MirroredStrategy, all replicas should have the same result
    result_val = strategy.experimental_local_results(result)[0]

    # Assertions to verify correctness
    # Expected shape: (1, 65536)
    assert result_val.shape == (1, 32 * 2048), f"Shape mismatch: {result_val.shape}"

    # Verify values
    # Since x and y are zeros, the result should purely be the reshaped arange tensor
    # The arange is -2048 to -1 (2048 elements). It is tiled 32 times to fill 32*2048.
    expected_arange = tf.range(-2048, 0, dtype=tf.int32)
    expected_content = tf.tile(expected_arange, [32])
    expected_content = tf.reshape(expected_content, (1, 32 * 2048))

    # Check if the result matches the expected content
    is_correct = tf.reduce_all(tf.equal(result_val, expected_content))
    
    assert is_correct.numpy(), "Value mismatch: The computation did not produce the expected output."
    
    print("Test passed successfully.")

if __name__ == "__main__":
    test_mirrored_tensor_interleaving()