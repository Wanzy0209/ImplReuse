import torch
import tensorflow as tf

def test_traceback_filtering_state_change():
    """
    Test case adapted from the logic of Issue 162313.
    
    The original issue involved a function with conditional branches controlled by a global flag,
    where `torch._dynamo.graph_break` was called inside those branches. The bug manifested when
    the global flag changed between executions.
    
    This test mirrors that structure using `tf.debugging.is_traceback_filtering_enabled` as the
    global flag check. It verifies that the API correctly reflects state changes between
    function calls, ensuring the conditional logic behaves as expected.
    """
    # Ensure initial state is enabled (default behavior)
    tf.debugging.enable_traceback_filtering()

    # Define a function that branches based on the global flag,
    # mirroring the structure of the original bug's 'fn'.
    def conditional_behavior():
        if tf.debugging.is_traceback_filtering_enabled():
            # Corresponds to 'if flag:' block in the original bug
            return "filtering_on"
        else:
            # Corresponds to 'else:' block in the original bug
            return "filtering_off"

    # First execution: Flag is True
    result_1 = conditional_behavior()
    assert result_1 == "filtering_on", "Expected filtering to be enabled initially"

    # Change the global state (mimicking 'flag = False' in the original bug)
    tf.debugging.disable_traceback_filtering()

    # Second execution: Flag is now False
    # In the original bug, this path change caused a KeyError in resume_execution.
    # Here, we verify that the similar API correctly reports the new state.
    result_2 = conditional_behavior()
    assert result_2 == "filtering_off", "Expected filtering to be disabled after toggle"

    # Reset to default state for cleanup
    tf.debugging.enable_traceback_filtering()

if __name__ == "__main__":
    test_traceback_filtering_state_change()
    print("Test passed.")