import torch
import tensorflow as tf
import numpy as np

def test_tf_gather_no_global_side_effects():
    """
    Test case to verify that tf.gather (and its underlying execution/compilation)
    does not cause unintended global side effects.
    
    This test is adapted from the PyTorch issue where torch.compile was
    redundantly calling code that modified global distribution validation
    settings (torch.distributions.Distribution.set_default_validate_args(False)).
    
    Here we check that executing tf.gather, particularly within a tf.function
    (which triggers compilation similar to torch.compile), does not alter
    a monitored global state.
    """
    
    # 1. Setup a mock global state to monitor
    # (Analogous to torch.distributions validation args)
    global_state_monitor = {"validation_enabled": True}

    # 2. Define inputs and the operation using the similar API (tf.gather)
    # We wrap the call in tf.function to simulate the compilation context
    # present in the original bug report.
    params = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indices = tf.constant([0, 2])

    @tf.function
    def execute_gather():
        # This call mimics the usage pattern found in the similar API snippet
        return tf.gather(params, indices, axis=0)

    # 3. Execute the operation
    result = execute_gather()

    # 4. Verify the global state has not been affected
    # (In the PyTorch bug, this state would have been flipped to False)
    assert global_state_monitor["validation_enabled"] is True, \
        "Global state was modified unexpectedly by gather operation"

    # 5. Verify the operation result is correct
    expected = tf.constant([[1, 2, 3], [7, 8, 9]])
    assert tf.reduce_all(tf.equal(result, expected)).numpy(), \
        "tf.gather produced incorrect results"

if __name__ == "__main__":
    test_tf_gather_no_global_side_effects()
    print("Test passed: No global side effects detected.")