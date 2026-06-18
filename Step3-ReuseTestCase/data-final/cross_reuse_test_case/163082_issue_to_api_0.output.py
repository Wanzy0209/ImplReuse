import torch
import tensorflow as tf
import numpy as np

def test_squared_hinge_precision():
    """
    Test case adapted from the logic of PyTorch Issue 163082.
    The original issue reported that torch.nn.functional.normalize produced
    vectors with norm > 1 when using torch.compile on CUDA.
    
    This test applies a similar logic to tf.keras.losses.squared_hinge:
    It compares the output of the loss function when executed via 
    tf.function (graph mode/compiled) vs eager execution to detect 
    potential numerical discrepancies or precision issues.
    """
    
    # Mimic the @torch.compile() decorator using tf.function
    @tf.function
    def loss_compiled(y_true, y_pred):
        return tf.keras.losses.squared_hinge(y_true, y_pred)

    # Define the eager (non-compiled) version
    def loss_eager(y_true, y_pred):
        return tf.keras.losses.squared_hinge(y_true, y_pred)

    # Setup inputs similar to the bug report (float32, specific values)
    # Using values close to the hinge boundary (1.0) to stress precision
    y_true = tf.constant([[1.0]], dtype=tf.float32)
    y_pred = tf.constant([[0.999999]], dtype=tf.float32)

    print("Input y_true:", y_true.numpy().flatten())
    print("Input y_pred:", y_pred.numpy().flatten())

    # Execute compiled version
    result_compiled = loss_compiled(y_true, y_pred)
    print("Loss (compiled):", result_compiled.numpy().flatten())

    # Execute eager version
    result_eager = loss_eager(y_true, y_pred)
    print("Loss (eager):", result_eager.numpy().flatten())

    # Assert that the results are consistent.
    # The original bug found a deviation (norm > 1.0). Here we check for any
    # significant deviation between compiled and eager modes.
    diff = np.abs(result_compiled.numpy() - result_eager.numpy())
    assert np.all(diff < 1e-6), \
        f"Numerical discrepancy detected between compiled and eager execution: {diff}"

if __name__ == "__main__":
    test_squared_hinge_precision()