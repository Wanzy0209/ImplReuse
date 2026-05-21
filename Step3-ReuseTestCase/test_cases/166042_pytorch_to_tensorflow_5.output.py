import torch
import tensorflow as tf
import numpy as np

def test_map_and_batch_bfloat16_ops():
    """
    Adapts the PyTorch fuzzer logic to TensorFlow's tf.data.experimental.map_and_batch.
    The original PyTorch issue involved a sequence of bfloat16 matrix multiplications
    and additions leading to a type assertion failure in an embedding operation.
    This test verifies that the TensorFlow data pipeline can handle the equivalent
    complex bfloat16 operations within the map_and_batch transformation.
    """

    # Define the map function that mimics the logic of the PyTorch fuzzed_program
    def fuzzed_map_func(arg_0):
        # Cast input to bfloat16 to match PyTorch: var_node_4 = arg_0 # dtype=bfloat16
        var_node_4 = tf.cast(arg_0, tf.bfloat16)

        # var_node_5 = torch.full((8, 7), -0.80078125, dtype=torch.bfloat16)
        var_node_5 = tf.fill([8, 7], tf.constant(-0.80078125, dtype=tf.bfloat16))

        # var_node_3 = torch.matmul(var_node_4, var_node_5)
        var_node_3 = tf.matmul(var_node_4, var_node_5)

        # Simulating other arguments (arg_1, arg_2) as constant weights for the map function
        # var_node_7 = arg_1 # size=(7, 12), dtype=bfloat16
        var_node_7 = tf.constant(np.random.randn(7, 12), dtype=tf.bfloat16)
        # var_node_8 = arg_2 # size=(12, 2), dtype=bfloat16
        var_node_8 = tf.constant(np.random.randn(12, 2), dtype=tf.bfloat16)

        # var_node_6 = torch.matmul(var_node_7, var_node_8)
        var_node_6 = tf.matmul(var_node_7, var_node_8)

        # var_node_2 = torch.matmul(var_node_3, var_node_6)
        var_node_2 = tf.matmul(var_node_3, var_node_6)

        # var_node_11 = torch.full((2, 3), 1.515625, dtype=torch.bfloat16)
        var_node_11 = tf.fill([2, 3], tf.constant(1.515625, dtype=tf.bfloat16))
        # var_node_12 = torch.full((3, 16), 0.2353515625, dtype=torch.bfloat16)
        var_node_12 = tf.fill([3, 16], tf.constant(0.2353515625, dtype=tf.bfloat16))

        # var_node_10 = torch.matmul(var_node_11, var_node_12)
        var_node_10 = tf.matmul(var_node_11, var_node_12)

        # var_node_14 = torch.full((16, 4), 2.21875, dtype=torch.bfloat16)
        var_node_14 = tf.fill([16, 4], tf.constant(2.21875, dtype=tf.bfloat16))
        # var_node_15 = torch.full((4, 9), -1.7421875, dtype=torch.bfloat16)
        var_node_15 = tf.fill([4, 9], tf.constant(-1.7421875, dtype=tf.bfloat16))

        # var_node_13 = torch.matmul(var_node_14, var_node_15)
        var_node_13 = tf.matmul(var_node_14, var_node_15)

        # var_node_9 = torch.matmul(var_node_10, var_node_13)
        var_node_9 = tf.matmul(var_node_10, var_node_13)

        # var_node_1 = torch.matmul(var_node_2, var_node_9)
        var_node_1 = tf.matmul(var_node_2, var_node_9)

        # var_node_19 = arg_3 # size=(14, 9), dtype=bfloat16
        var_node_19 = tf.constant(np.random.randn(14, 9), dtype=tf.bfloat16)
        # var_node_20 = torch.full((9, 2), 0.8203125, dtype=torch.bfloat16)
        var_node_20 = tf.fill([9, 2], tf.constant(0.8203125, dtype=tf.bfloat16))

        # var_node_18 = torch.matmul(var_node_19, var_node_20)
        var_node_18 = tf.matmul(var_node_19, var_node_20)

        # var_node_22 = arg_4 # size=(2,), dtype=bfloat16
        var_node_22 = tf.constant(np.random.randn(2), dtype=tf.bfloat16)
        # var_node_23 = torch.full((2,), -0.7421875, dtype=torch.bfloat16)
        var_node_23 = tf.fill([2], tf.constant(-0.7421875, dtype=tf.bfloat16))

        # var_node_21 = torch.add(var_node_22, var_node_23)
        var_node_21 = tf.add(var_node_22, var_node_23)

        # var_node_17 = torch.matmul(var_node_18, var_node_21)
        var_node_17 = tf.matmul(var_node_18, var_node_21)

        # Return the computed tensors to be batched
        return var_node_1, var_node_17

    # Create a dataset with inputs matching arg_0 shape (4, 8)
    # Using float32 inputs which will be cast to bfloat16 inside the map function
    dummy_data = [np.random.randn(4, 8).astype(np.float32) for _ in range(10)]
    dataset = tf.data.Dataset.from_tensor_slices(dummy_data)

    # Apply map_and_batch
    # This tests the pipeline's ability to handle the bfloat16 operations derived from the bug report
    batch_size = 4
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            map_func=fuzzed_map_func,
            batch_size=batch_size,
            drop_remainder=True,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
    )

    # Verify execution and output properties
    print("Testing tf.data.experimental.map_and_batch with adapted bfloat16 logic...")
    for batch_var_1, batch_var_17 in dataset.take(1):
        # Check shapes
        # var_node_1 is (4, 9), batched -> (4, 4, 9)
        assert batch_var_1.shape == (batch_size, 4, 9), f"Expected shape (4, 4, 9), got {batch_var_1.shape}"
        # var_node_17 is (14,), batched -> (4, 14)
        assert batch_var_17.shape == (batch_size, 14), f"Expected shape (4, 14), got {batch_var_17.shape}"

        # Check dtypes
        assert batch_var_1.dtype == tf.bfloat16, f"Expected dtype bfloat16, got {batch_var_1.dtype}"
        assert batch_var_17.dtype == tf.bfloat16, f"Expected dtype bfloat16, got {batch_var_17.dtype}"

    print("Test passed: map_and_batch handled the bfloat16 operations successfully.")

if __name__ == "__main__":
    test_map_and_batch_bfloat16_ops()