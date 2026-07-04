import torch
import tensorflow as tf
import numpy as np

def test_relu_no_global_side_effects():
    """
    Test that tf.keras.activations.relu does not modify global Keras settings.
    
    This test is based on the logic of Issue #167064, where torch.compile 
    (called internally) unexpectedly modified global distribution settings 
    (torch.distributions.Distribution.set_default_validate_args). 
    
    Here, we verify that calling tf.keras.activations.relu (which internally 
    retrieves a default context) does not similarly pollute global state 
    such as the Keras floatx or image data format.
    """
    # 1. Capture and set a specific global state to test against
    original_floatx = tf.keras.backend.floatx()
    target_floatx = 'float16' if original_floatx == 'float32' else 'float32'
    
    tf.keras.backend.set_floatx(target_floatx)
    
    # 2. Call the API under test
    # The similar API implementation shows it uses context.get_default().
    # We ensure the function executes correctly.
    input_tensor = tf.constant([-1.0, 0.0, 1.0])
    output = tf.keras.activations.relu(input_tensor)
    
    # 3. Verify that the global state remains unchanged
    # (Analogous to checking if torch.compile changed distribution validation)
    current_floatx = tf.keras.backend.floatx()
    assert current_floatx == target_floatx, (
        f"Global floatx setting was unexpectedly changed from {target_floatx} "
        f"to {current_floatx} by tf.keras.activations.relu"
    )
    
    # 4. Verify the output is correct (sanity check)
    expected_output = np.array([0.0, 0.0, 1.0], dtype=np.float16 if target_floatx == 'float16' else np.float32)
    np.testing.assert_array_almost_equal(output.numpy(), expected_output)

    # Cleanup
    tf.keras.backend.set_floatx(original_floatx)

if __name__ == "__main__":
    test_relu_no_global_side_effects()
    print("Test passed: tf.keras.activations.relu does not affect global state.")