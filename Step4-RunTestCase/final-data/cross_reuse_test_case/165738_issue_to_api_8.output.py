import tensorflow as tf
import numpy as np

# Enable eager execution to support .numpy() method on tensors.
# This is necessary because the test environment might be running in graph mode (TF 1.x style)
# where .numpy() is not available on Tensor objects.
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

def test_floatx_configuration_and_sqrt():
    """
    Test case derived from Issue 165738 regarding performance of sqrt operations 
    with specific float types (fp16) on XPU.
    
    This test verifies the behavior of the similar API `tf.keras.backend.floatx()`
    in the context of setting and using the default float type for mathematical 
    operations, mirroring the environment setup in the original bug report.
    """
    # Save original floatx to restore later
    original_floatx = tf.keras.backend.floatx()

    try:
        # The original bug report highlights issues with fp16 ('*fp16' in signature).
        # We set the Keras backend to use float16 to simulate this precision context.
        tf.keras.backend.set_floatx('float16')
        
        # Verify the API returns the configured type
        current_floatx = tf.keras.backend.floatx()
        assert current_floatx == 'float16', \
            f"Expected floatx to be 'float16', but got '{current_floatx}'"

        # Perform a sqrt operation (the operation associated with the bug report)
        # using the default float type configured by the API under test.
        # Note: While we cannot test XPU performance regression here, we ensure
        # the type configuration logic for the operation is correct.
        input_data = tf.constant([4.0, 9.0, 16.0], dtype=current_floatx)
        result = tf.sqrt(input_data)

        # Assert the operation runs and produces correct results in the configured precision
        expected = tf.constant([2.0, 3.0, 4.0], dtype=current_floatx)
        np.testing.assert_allclose(result.numpy(), expected.numpy(), rtol=1e-3)

    finally:
        # Restore original floatx to avoid side effects
        tf.keras.backend.set_floatx(original_floatx)

if __name__ == "__main__":
    test_floatx_configuration_and_sqrt()
    print("Test passed.")