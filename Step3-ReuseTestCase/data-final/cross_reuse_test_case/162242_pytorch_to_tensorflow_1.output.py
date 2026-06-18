import numpy as np
import tensorflow as tf


def test_argsort_determinism():
    """
    Adapted test to check determinism of tf.experimental.numpy.argsort 
    with duplicate values (analogous to duplicated indices in scatter).
    """
    # Create input data with duplicate values to test sorting stability/determinism
    # Shape (2, 3, 4) to match the dimensionality of the original test
    input_data = np.array(
        [
            [
                [1.0, 0.0, 1.0, 2.0],
                [2.0, 2.0, 0.0, 1.0],
                [0.0, 1.0, 2.0, 2.0],
            ],
            [
                [3.0, 3.0, 1.0, 0.0],
                [1.0, 2.0, 1.0, 3.0],
                [0.0, 0.0, 1.0, 1.0],
            ],
        ],
        dtype=np.float32,
    )

    # Determine the device to use (GPU if available, mimicking the original .cuda())
    device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

    # Run once to establish the ground truth result
    # The original test observed deterministic behavior contrary to docs.
    # We check if argsort behaves consistently here.
    with tf.device(device):
        gt_res = tf.experimental.numpy.argsort(input_data, axis=-1, kind='quicksort').numpy()

    for i in range(1000):
        with tf.device(device):
            # tf.experimental.numpy.argsort wraps sort_ops.argsort
            # kind='quicksort' maps to stable=False in the implementation
            res = tf.experimental.numpy.argsort(input_data, axis=-1, kind='quicksort')

        print(f"Test {i + 1}/{1000}")
        # Assert that the result matches the ground truth (determinism check)
        np.testing.assert_array_equal(res.numpy(), gt_res)


if __name__ == "__main__":
    test_argsort_determinism()