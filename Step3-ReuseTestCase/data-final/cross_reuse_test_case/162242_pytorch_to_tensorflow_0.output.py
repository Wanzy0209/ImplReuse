import torch
import numpy as np
import tensorflow as tf


def test_argsort_determinism():
    # Create a tensor with duplicate values to test determinism
    # Shape (2, 3, 4) similar to original test
    # Duplicates are introduced to potentially trigger non-deterministic behavior in unstable sorts
    data = np.array(
        [
            [
                [1.0, 3.0, 2.0, 3.0],
                [0.0, 0.0, 1.0, 2.0],
                [5.0, 4.0, 5.0, 6.0],
            ],
            [
                [10.0, 10.0, 12.0, 11.0],
                [9.0, 8.0, 9.0, 7.0],
                [15.0, 14.0, 13.0, 13.0],
            ],
        ],
        dtype=np.float32,
    )

    inputs = tf.constant(data)

    # Run once to establish the "observed" ground truth.
    # This mimics the user observing a result and checking if it changes over time.
    # We use axis=1 to match the dimension used in the original torch.scatter call.
    gt_res = tf.argsort(inputs, axis=1, kind="quicksort").numpy()

    for i in range(1000):
        # tf.argsort
        # axis=1 matches the 'dim=1' in the original torch.scatter
        res = tf.argsort(inputs, axis=1, kind="quicksort")

        print(f"Test {i + 1}/{1000}")
        # Assert that the result is deterministic and matches the first observed result
        np.testing.assert_array_equal(res.numpy(), gt_res)


if __name__ == "__main__":
    test_argsort_determinism()