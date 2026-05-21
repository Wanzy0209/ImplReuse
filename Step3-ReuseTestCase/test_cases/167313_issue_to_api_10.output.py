import torch
import tensorflow as tf
import numpy as np

def test_squared_hinge_coefficients():
    """
    Test case to verify that tf.keras.losses.squared_hinge correctly handles
    the internal coefficients (specifically the constant '1' in the hinge formula)
    in both eager and graph (compiled) modes.

    This mirrors the logic of the PyTorch bug where torch.compile (Inductor)
    ignored alpha/beta parameters in addmm. Here we verify that the '1' in
    '1 - y_true * y_pred' is not ignored during graph tracing/compilation.
    """
    # Setup inputs
    # y_true values are expected to be -1 or 1. 
    # We use a fixed seed for reproducibility.
    np.random.seed(42)
    y_true = np.random.choice([-1, 1], size=(2, 3)).astype(np.float32)
    y_pred = np.random.rand(2, 3).astype(np.float32)

    # 1. Calculate loss using the API in Eager mode
    loss_eager = tf.keras.losses.squared_hinge(y_true, y_pred)

    # 2. Calculate loss using the API in Compiled (Graph) mode
    # This is analogous to torch.compile in the original bug report.
    @tf.function
    def compiled_loss(y_t, y_p):
        return tf.keras.losses.squared_hinge(y_t, y_p)

    loss_compiled = compiled_loss(y_true, y_pred)

    # 3. Calculate the expected result manually based on the definition:
    # loss = mean(square(maximum(1 - y_true * y_pred, 0)), axis=-1)
    # We explicitly check the '1' constant here, similar to checking alpha/beta.
    y_true_t = tf.constant(y_true)
    y_pred_t = tf.constant(y_pred)
    
    # Manual implementation of the formula
    expected_loss = tf.reduce_mean(
        tf.square(
            tf.maximum(1.0 - y_true_t * y_pred_t, 0.0)
        ),
        axis=-1
    )

    # Assertions
    # Check if compiled mode matches eager mode (catches optimization bugs)
    np.testing.assert_allclose(
        loss_eager.numpy(), 
        loss_compiled.numpy(), 
        rtol=1e-5, 
        err_msg="Compiled mode produced different results than Eager mode."
    )

    # Check if the result matches the mathematical definition (catches logic bugs)
    np.testing.assert_allclose(
        loss_eager.numpy(), 
        expected_loss.numpy(), 
        rtol=1e-5, 
        err_msg="API output does not match the mathematical definition (1 - y_true * y_pred)."
    )

    print("Test passed: squared_hinge preserves coefficients in graph mode.")

if __name__ == "__main__":
    test_squared_hinge_coefficients()