import torch
import tensorflow as tf
import numpy as np

# Test case adapted from Issue 164143: DebugMode silently disables torch.compile
# Replaced torch.compile,DebugMode with tf.keras.initializers.LecunUniform
# Leveraging torch.nn.functional.tanh (Similar API) -> tf.nn.tanh

def test_lecun_uniform_behavior():
    """
    Test that tf.keras.initializers.LecunUniform works correctly and does not
    silently disable or fail, analogous to the original bug report where
    torch.compile would skip execution in DebugMode.
    """
    # Replacement API: tf.keras.initializers.LecunUniform
    initializer = tf.keras.initializers.LecunUniform(seed=42)

    # Define a shape
    shape = (10, 10)
    fan_in = float(shape[1])
    limit = np.sqrt(3.0 / fan_in)

    # Execute the initializer
    # In the original bug, execution was skipped. Here we ensure it happens.
    weights = initializer(shape)

    # Assertion 1: Ensure it didn't silently return None or skip
    assert weights is not None, "Initializer silently disabled (returned None)"
    assert weights.shape == shape, f"Shape mismatch: expected {shape}, got {weights.shape}"

    # Assertion 2: Verify the mathematical properties (LecunUniform bounds)
    # LecunUniform samples from U[-limit, limit]
    # We check if values are within bounds.
    assert tf.reduce_all(weights >= -limit), "Values found below lower bound"
    assert tf.reduce_all(weights <= limit), "Values found above upper bound"

    # Assertion 3: Leverage the similar API (tanh)
    # LecunUniform is designed to work with tanh activations.
    # We verify the pipeline works.
    activations = tf.nn.tanh(weights)
    assert activations is not None, "Tanh operation failed silently"
    assert activations.shape == shape, "Tanh output shape mismatch"

    print("Test passed: LecunUniform executed correctly without silent failure.")

if __name__ == "__main__":
    test_lecun_uniform_behavior()