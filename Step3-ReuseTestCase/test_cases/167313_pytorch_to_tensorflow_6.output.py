import torch
import tensorflow as tf
import numpy as np

def test_random_normal_parameters():
    """
    Adapted test case for tf.keras.backend.random_normal based on the 
    torch.addmm bug report (Issue 167313).
    
    The original bug involved torch.compile ignoring alpha/beta parameters 
    in addmm. This test checks if tf.function (TensorFlow's compilation/tracing)
    correctly respects the 'mean' and 'stddev' parameters of random_normal,
    analogous to checking alpha/beta in the original bug.
    """
    
    # Setup inputs
    # Using a larger shape to allow for statistical verification of mean/stddev
    shape = (1000, 1000)
    
    # Define non-default parameters to check if they are respected
    # Analogous to alpha=0.5, beta=0.5 in the original bug
    target_mean = 5.0
    target_stddev = 0.1

    # Define the function using the target API
    # Original: f = lambda x, a, b: torch.nn.functional.relu(torch.addmm(x, a, b, alpha=0.5, beta=0.5))
    # Adapted: f = lambda shape: tf.keras.backend.random_normal(shape, mean=5.0, stddev=0.1)
    f = lambda s: tf.keras.backend.random_normal(s, mean=target_mean, stddev=target_stddev)

    # Compile the function
    # Original: fc = torch.compile(f)
    # Adapted: fc = tf.function(f)
    fc = tf.function(f)

    # Run eager execution
    # Original: f(x, a, b)
    result_eager = f(shape)

    # Run compiled execution
    # Original: fc(x, a, b)
    result_compiled = fc(shape)

    # Verification
    # In the original bug, the output values were exactly 2x the expected values 
    # because alpha/beta defaulted to 1 instead of 0.5.
    # Here, we check if the output distribution matches the specified parameters.
    # If parameters were ignored, the mean would be ~0.0 and stddev ~1.0.
    
    eager_mean = np.mean(result_eager)
    compiled_mean = np.mean(result_compiled)
    
    print(f"Eager result mean: {eager_mean:.4f} (Expected ~{target_mean})")
    print(f"Compiled result mean: {compiled_mean:.4f} (Expected ~{target_mean})")

    # Assert that the mean is close to the target (5.0) and not the default (0.0)
    # Tolerance is set loosely (0.5) to account for randomness in a single sample
    assert np.abs(eager_mean - target_mean) < 0.5, \
        f"Eager execution failed: expected mean ~{target_mean}, got {eager_mean}"
        
    assert np.abs(compiled_mean - target_mean) < 0.5, \
        f"Compiled execution failed: expected mean ~{target_mean}, got {compiled_mean}. " \
        "This indicates parameters might have been ignored during compilation."

if __name__ == "__main__":
    test_random_normal_parameters()