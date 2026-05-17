import torch
import tensorflow as tf
import numpy as np

def test_cumsum_gradient_behavior():
    """
    Test case adapted from PyTorch issue #160273 regarding gradient behavior.
    
    The original issue highlights that torch.min behaves differently regarding 
    gradients depending on whether it reduces over all dimensions (distributing 
    gradients) or a specific dimension (indexing-like).
    
    This test verifies the gradient behavior of tf.keras.ops.cumsum to ensure
    gradients are distributed correctly along the specified axis, analogous to
    checking the gradient flow in the original issue.
    """
    
    # --- Case 1: Operation on 1D tensor (Analogous to 'reduce over all dimensions') ---
    # Setup: Create a tensor of ones
    x = tf.ones([5])
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        # Operation: Cumulative sum
        # Note: Unlike torch.min, cumsum preserves shape. To mimic the scalar 
        # backward() in the original issue, we reduce the output to a scalar.
        y = tf.keras.ops.cumsum(x, axis=0)
        loss = tf.reduce_sum(y)
    
    # Backward: Compute gradients
    grads = tape.gradient(loss, x)
    
    # Assertion: 
    # For x=[1, 1, 1, 1, 1], cumsum is [1, 2, 3, 4, 5]. Sum is 15.
    # d(Sum)/dx[i] is the number of times x[i] appears in the cumulative sums.
    # x[0] appears in all 5 sums -> grad 5
    # x[4] appears in 1 sum -> grad 1
    # Expected: [5, 4, 3, 2, 1]
    expected_grads_1d = tf.constant([5., 4., 3., 2., 1.])
    assert np.allclose(grads.numpy(), expected_grads_1d.numpy()), \
        f"1D Gradient mismatch. Expected {expected_grads_1d.numpy()}, got {grads.numpy()}"

    # --- Case 2: Operation on 2D tensor with specific axis (Analogous to 'reduce over specified dimension') ---
    # Setup: Create a 2D tensor of ones
    x_2d = tf.ones([2, 3])
    
    with tf.GradientTape() as tape:
        tape.watch(x_2d)
        # Operation: Cumulative sum along axis=1
        y_2d = tf.keras.ops.cumsum(x_2d, axis=1)
        loss_2d = tf.reduce_sum(y_2d)
        
    grads_2d = tape.gradient(loss_2d, x_2d)
    
    # Assertion:
    # Shape is (2, 3). Axis 1 has length 3.
    # Row 0: [1, 1, 1] -> cumsum -> [1, 2, 3] -> sum -> 6.
    # Gradients for Row 0: [3, 2, 1] (x[0,0] used 3 times, x[0,1] used 2 times, etc.)
    # Row 1 is identical.
    expected_grads_2d = tf.constant([[3., 2., 1.], 
                                     [3., 2., 1.]])
    assert np.allclose(grads_2d.numpy(), expected_grads_2d.numpy()), \
        f"2D Gradient mismatch. Expected {expected_grads_2d.numpy()}, got {grads_2d.numpy()}"

    print("Test passed: tf.keras.ops.cumsum gradient behavior is consistent.")

if __name__ == "__main__":
    test_cumsum_gradient_behavior()