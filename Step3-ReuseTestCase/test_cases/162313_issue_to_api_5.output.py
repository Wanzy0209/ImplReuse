import torch
import tensorflow as tf

def test_traceback_filtering_state_change_in_tf_function():
    """
    Test case adapted from Issue 162313.
    
    Original Bug Pattern:
    - A compiled function (@torch.compile) uses a global flag to branch logic.
    - The flag is toggled between calls.
    - This caused a KeyError in PyTorch's resume execution logic.

    Adaptation for Similar API (tf.keras.config.is_traceback_filtering_enabled):
    - Use @tf.function (equivalent to @torch.compile).
    - Use tf.keras.config.is_traceback_filtering_enabled() (the similar API) 
      to determine control flow, replacing the manual 'flag' variable.
    - Toggle the traceback filtering state between calls to test 
      tracing/execution consistency.
    """
    
    # Ensure initial state is enabled
    tf.debugging.enable_traceback_filtering()

    @tf.function
    def fn(x):
        # Using the similar API to control logic flow
        # This mirrors the 'if flag:' structure in the original bug report
        if tf.keras.config.is_traceback_filtering_enabled():
            # Branch 1: Filtering enabled
            return x + 1
        else:
            # Branch 2: Filtering disabled
            return x + 2

    # First call with filtering enabled
    result_1 = fn(tf.constant(1))
    assert result_1.numpy() == 2, "First call failed: expected 2 (1 + 1)"

    # Change the global state (mimics 'flag = False' in the original bug)
    tf.debugging.disable_traceback_filtering()

    # Second call with filtering disabled
    # In the original bug, this path change triggered a KeyError.
    # Here we verify the function correctly handles the state change.
    result_2 = fn(tf.constant(1))
    assert result_2.numpy() == 3, "Second call failed: expected 3 (1 + 2)"

    # Restore global state for cleanliness
    tf.debugging.enable_traceback_filtering()
    
    print("Test passed successfully.")

if __name__ == "__main__":
    test_traceback_filtering_state_change_in_tf_function()