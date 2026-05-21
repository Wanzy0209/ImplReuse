import torch
import tensorflow as tf
import numpy as np

def test_kl_divergence_compilation():
    """
    Test case for tf.keras.losses.kullback_leibler_divergence (related to 
    tf.compat.v1.distributions.kl_divergence) to ensure compilation 
    (tf.function) does not ignore parameters or produce incorrect results,
    mirroring the logic of the PyTorch addmm bug report.
    """
    # Setup data
    # KL divergence requires positive values for log operations
    np.random.seed(42)
    y_true = np.random.rand(2, 3).astype(np.float32) + 0.1
    y_pred = np.random.rand(2, 3).astype(np.float32) + 0.1

    # Define the function using the similar API
    # We use a reduction parameter to mimic the 'alpha/beta' parameter usage in the original bug
    # and wrap it in a pointwise operation (relu) to mimic the trigger condition.
    f = lambda y, p: tf.nn.relu(
        tf.keras.losses.kullback_leibler_divergence(y, p, reduction=tf.keras.losses.Reduction.SUM)
    )

    # Eager execution
    result_eager = f(y_true, y_pred)

    # Compiled execution (tf.function is the TF equivalent of torch.compile)
    @tf.function
    def fc(y, p):
        return f(y, p)

    result_compiled = fc(y_true, y_pred)

    # Assert that compiled results match eager results
    # The original bug showed a discrepancy (e.g. 0.74 vs 1.48).
    # This test ensures the TF API handles the pattern correctly.
    np.testing.assert_allclose(result_eager.numpy(), result_compiled.numpy(), rtol=1e-5)
    print("Test passed: Eager and compiled results match.")

if __name__ == "__main__":
    test_kl_divergence_compilation()