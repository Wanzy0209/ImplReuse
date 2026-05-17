import torch
import tensorflow as tf
import numpy as np

def test_full_like_gradient_behavior():
    """
    Adapted test case based on the torch.min gradient issue.
    
    Original Issue: torch.min has different gradient behaviors depending on 
    whether it reduces over all dimensions (distribute evenly) or a specific 
    dimension (index-like).
    
    Adaptation for tf.experimental.numpy.full_like:
    Since full_like creates a tensor filled with a constant value based on the 
    shape of the input, the output values do not depend on the input values. 
    Therefore, the gradient with respect to the input tensor should be zero.
    """
    
    # Setup: Create a tensor similar to the PyTorch example (torch.ones([5]))
    # We use tf.Variable to track gradients
    a = tf.Variable(np.ones([5]), dtype=tf.float32)

    with tf.GradientTape() as tape:
        # Operation: tf.experimental.numpy.full_like
        # This creates a new tensor with the same shape as 'a', filled with 'fill_value'.
        # It does not perform a reduction like min, but we verify its gradient flow.
        fill_val = 5.0
        result = tf.experimental.numpy.full_like(a, fill_value=fill_val)

    # Backward pass: Calculate gradients of the result w.r.t 'a'
    grads = tape.gradient(result, a)

    # Verification
    # Expected behavior: The gradient should be all zeros because the operation
    # only uses the shape of 'a', not its data values.
    expected_grads = np.zeros([5])

    print("Input:", a.numpy())
    print("Output:", result.numpy())
    print("Gradients:", grads.numpy())

    # Assert that gradients are zero (or None if disconnected, though tape usually returns 0 here)
    if grads is not None:
        assert np.allclose(grads.numpy(), expected_grads), \
            f"Expected gradients to be zero, got {grads.numpy()}"
    else:
        # In some TF contexts, disconnected gradients might be None, 
        # but for shape operations on Variables, it's often zeros.
        print("Gradients are None (disconnected)")

if __name__ == "__main__":
    test_full_like_gradient_behavior()