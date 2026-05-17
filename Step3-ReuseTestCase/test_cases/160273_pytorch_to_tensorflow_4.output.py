import torch
import tensorflow as tf
import numpy as np

def test_tf_geomspace_gradients():
    """
    Adapted test case for tf.experimental.numpy.geomspace based on the 
    torch.min gradient behavior issue.
    
    The original issue highlighted inconsistent gradient distribution 
    depending on reduction dimensions. For geomspace, we verify that 
    gradients flow correctly to the input parameters (start, stop) 
    and check behavior across different parameter configurations.
    """
    
    print("--- Test Case 1: Basic Gradient Flow (endpoint=True) ---")
    # Define inputs
    start = tf.constant(1.0)
    stop = tf.constant(32.0)
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(start)
        tape.watch(stop)
        
        # Operation: geomspace(start, stop, num=5)
        # Expected sequence: [1, 2, 4, 8, 16, 32] (powers of 2)
        vals = tf.experimental.numpy.geomspace(start, stop, num=6, endpoint=True)
        
        # To perform backward pass, we need a scalar loss. 
        # We sum the values, similar to how torch.min reduces to a scalar.
        loss = tf.reduce_sum(vals)

    # Compute gradients
    grad_start = tape.gradient(loss, start)
    grad_stop = tape.gradient(loss, stop)

    print(f"Input start: {start.numpy()}, Input stop: {stop.numpy()}")
    print(f"Output values: {vals.numpy()}")
    print(f"Gradient w.r.t start: {grad_start.numpy()}")
    print(f"Gradient w.r.t stop: {grad_stop.numpy()}")
    
    # Basic sanity check: gradients should not be None or NaN
    assert grad_start is not None, "Gradient w.r.t start should not be None"
    assert grad_stop is not None, "Gradient w.r.t stop should not be None"
    assert not np.isnan(grad_start.numpy()), "Gradient w.r.t start is NaN"
    assert not np.isnan(grad_stop.numpy()), "Gradient w.r.t stop is NaN"

    print("\n--- Test Case 2: Gradient Flow with endpoint=False ---")
    # Check if changing the 'endpoint' argument affects gradient calculation logic
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(start)
        tape.watch(stop)
        
        # Operation: geomspace(start, stop, num=5, endpoint=False)
        # Expected sequence: [1, 2, 4, 8, 16] (stop is exclusive)
        vals_no_end = tf.experimental.numpy.geomspace(start, stop, num=5, endpoint=False)
        loss_no_end = tf.reduce_sum(vals_no_end)

    grad_start_no_end = tape.gradient(loss_no_end, start)
    grad_stop_no_end = tape.gradient(loss_no_end, stop)

    print(f"Output values (no endpoint): {vals_no_end.numpy()}")
    print(f"Gradient w.r.t start (no endpoint): {grad_start_no_end.numpy()}")
    print(f"Gradient w.r.t stop (no endpoint): {grad_stop_no_end.numpy()}")
    
    assert grad_start_no_end is not None, "Gradient w.r.t start should not be None"
    assert grad_stop_no_end is not None, "Gradient w.r.t stop should not be None"

    print("\n--- Test Case 3: Gradient Flow with axis parameter ---")
    # The original torch.min issue showed different behavior for 'dim'.
    # geomspace has an 'axis' parameter. We verify gradients are consistent 
    # regardless of axis (as axis only affects output layout, not the math).
    
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(start)
        tape.watch(stop)
        
        # axis=-1
        vals_axis = tf.experimental.numpy.geomspace(start, stop, num=6, endpoint=True, axis=-1)
        loss_axis = tf.reduce_sum(vals_axis)

    grad_start_axis = tape.gradient(loss_axis, start)
    grad_stop_axis = tape.gradient(loss_axis, stop)

    print(f"Gradient w.r.t start (axis=-1): {grad_start_axis.numpy()}")
    print(f"Gradient w.r.t stop (axis=-1): {grad_stop_axis.numpy()}")
    
    # Gradients should be identical to Test Case 1 because the math on start/stop is the same
    assert np.allclose(grad_start.numpy(), grad_start_axis.numpy()), \
        "Gradient w.r.t start should be independent of axis parameter"
    assert np.allclose(grad_stop.numpy(), grad_stop_axis.numpy()), \
        "Gradient w.r.t stop should be independent of axis parameter"

if __name__ == "__main__":
    test_tf_geomspace_gradients()