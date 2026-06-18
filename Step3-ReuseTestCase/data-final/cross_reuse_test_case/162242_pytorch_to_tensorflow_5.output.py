import torch
import numpy as np
import tensorflow as tf


def test_normalize_deterministic():
    """
    Adapted test case for tf.keras.ops.normalize to verify determinism,
    similar to the logic used in the torch.scatter issue report.
    """
    # Setup input data mimicking the shape and type of the original test
    inputs = tf.range(24, dtype=tf.float32)
    inputs = tf.reshape(inputs, [2, 3, 4])

    # Run once to establish ground truth
    with tf.GradientTape() as tape:
        tape.watch(inputs)
        res = tf.keras.ops.normalize(inputs, axis=-1, order=2)
    
    gt_res = res.numpy()
    gt_grad = tape.gradient(res, inputs).numpy()

    # Run multiple times to check for non-determinism
    for i in range(1000):
        with tf.GradientTape() as tape:
            tape.watch(inputs)
            res = tf.keras.ops.normalize(inputs, axis=-1, order=2)
        
        grad = tape.gradient(res, inputs).numpy()

        if (i + 1) % 100 == 0:
            print(f"Test {i + 1}/1000")

        # Assert that the result and gradients are deterministic
        np.testing.assert_allclose(res.numpy(), gt_res)
        np.testing.assert_allclose(grad, gt_grad)


if __name__ == "__main__":
    test_normalize_deterministic()