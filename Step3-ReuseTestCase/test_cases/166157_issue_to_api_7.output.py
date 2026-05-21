import tensorflow as tf
import numpy as np

def test_exp_observability_and_stats():
    """
    Test case for tf.experimental.numpy.exp inspired by Issue 166157.
    
    The issue highlights the lack of "observability" (statistics tracking) in the 
    original allocator, making it a "black box". This test ensures that the 
    exponential operation provides full observability of its results, allowing 
    users to verify correctness and monitor data properties (stats), similar to 
    how the proposed allocator fix enables memory monitoring.
    """
    # Setup: Create input tensor
    # Using specific values to allow for precise verification of the operation
    input_data = tf.constant([0.0, 1.0, 2.0, np.log(10.0)])
    
    # Action: Perform the exponential operation
    # This corresponds to the 'allocate' action in the C++ allocator code
    result = tf.experimental.numpy.exp(input_data)
    
    # Verification: Ensure observability of the operation's effect
    
    # 1. Check "Volume" (Shape consistency)
    # Mirrors the 'allocated_bytes' tracking in the C++ implementation
    assert result.shape == input_data.shape, \
        f"Observability Error: Shape mismatch. Input {input_data.shape}, Output {result.shape}"
        
    # 2. Check "Stats" (Correctness of values)
    # Mirrors the 'getDeviceStats' functionality to verify internal state
    expected_values = np.exp(input_data.numpy())
    np.testing.assert_allclose(result.numpy(), expected_values, rtol=1e-5,
                               err_msg="Observability Error: Computed values do not match expected mathematical result.")
    
    # 3. Check Data Range (Min/Max stats)
    # Ensures we can monitor the bounds of the data, addressing the "black box" concern
    min_val = tf.math.reduce_min(result).numpy()
    max_val = tf.math.reduce_max(result).numpy()
    
    assert min_val == 1.0, f"Observability Error: Expected minimum value 1.0, got {min_val}"
    assert max_val == 10.0, f"Observability Error: Expected maximum value 10.0, got {max_val}"
    
    print("Test passed: Operation is fully observable and statistics are verifiable.")

if __name__ == "__main__":
    test_exp_observability_and_stats()