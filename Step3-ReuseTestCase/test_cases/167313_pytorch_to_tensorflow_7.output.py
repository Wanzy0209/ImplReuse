import torch
import tensorflow as tf
import numpy as np

def test_random_uniform_params():
    """
    Adapted test case for tf.keras.backend.random_uniform based on the 
    torch.addmm bug report (Issue 167313).
    
    Original Bug: torch.compile ignored alpha/beta parameters in addmm, 
    defaulting them to 1.0.
    
    This test verifies that tf.keras.backend.random_uniform respects the 
    minval and maxval parameters in both eager and compiled (tf.function) modes,
    ensuring they are not silently replaced with defaults (0.0 and 1.0).
    """
    
    # Define specific parameters (analogous to alpha=0.5, beta=0.5 in the original bug)
    # Defaults for random_uniform are minval=0.0, maxval=1.0
    custom_minval = 5.0
    custom_maxval = 10.0
    shape = (100, 100)

    # Define the function using the API
    def f(shape, minv, maxv):
        return tf.keras.backend.random_uniform(shape, minval=minv, maxval=maxv)

    # Compile the function (analogous to torch.compile)
    fc = tf.function(f)

    # 1. Test Eager Execution
    result_eager = f(shape, custom_minval, custom_maxval)
    
    # Verify parameters are respected
    assert np.all(result_eager.numpy() >= custom_minval), \
        f"Eager mode failed: minval {custom_minval} ignored (found min {np.min(result_eager.numpy())})"
    assert np.all(result_eager.numpy() < custom_maxval), \
        f"Eager mode failed: maxval {custom_maxval} ignored (found max {np.max(result_eager.numpy())})"

    # 2. Test Compiled Execution
    result_compiled = fc(shape, custom_minval, custom_maxval)
    
    # Verify parameters are respected (checking for the specific bug type: params ignored)
    assert np.all(result_compiled.numpy() >= custom_minval), \
        f"Compiled mode failed: minval {custom_minval} ignored (found min {np.min(result_compiled.numpy())})"
    assert np.all(result_compiled.numpy() < custom_maxval), \
        f"Compiled mode failed: maxval {custom_maxval} ignored (found max {np.max(result_compiled.numpy())})"

    print("Test Passed: Parameters minval and maxval are correctly respected in both eager and compiled modes.")

if __name__ == "__main__":
    test_random_uniform_params()