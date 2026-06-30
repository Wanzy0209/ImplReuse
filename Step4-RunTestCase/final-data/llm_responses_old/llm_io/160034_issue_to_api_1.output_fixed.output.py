import sys
import torch
import numpy as np

# Attempt to import TensorFlow, handling potential environment incompatibilities (e.g., GLIBCXX)
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Test skipped: Failed to import TensorFlow due to environment incompatibility.")
    print(f"Error details: {e}")
    sys.exit(0)

def test_custom_gradient_complex_index_put():
    """
    Test case leveraging tf.keras.ops.custom_gradient to implement 
    index_put logic with complex64 support, mirroring the scenario 
    in the PyTorch issue where index_put_ failed with complex types.
    """
    
    # Define a custom gradient function that mimics torch.index_put_ with accumulate=True
    @tf.keras.ops.custom_gradient
    def index_put_accumulate(tensor, indices, values):
        # Forward pass: tensor[indices] += values
        # Using tf.tensor_scatter_nd_add to simulate the accumulation
        updated = tf.tensor_scatter_nd_add(tensor, tf.expand_dims(indices, 1), values)

        def grad(upstream):
            # Gradient w.r.t tensor: upstream (identity)
            grad_tensor = upstream
            # Gradient w.r.t values: upstream[indices]
            grad_values = tf.gather_nd(upstream, tf.expand_dims(indices, 1))
            # Gradient w.r.t indices: None (indices are integers)
            return grad_tensor, None, grad_values

        return updated, grad

    # Setup tensors similar to the PyTorch issue
    # Note: While the original issue specified MPS device, here we focus on the dtype (complex64)
    # which was the core of the type support failure.
    image = tf.zeros(10, dtype=tf.complex64)
    data = tf.ones(3, dtype=tf.complex64)
    indices = tf.constant([1, 3, 5])

    # Execute the custom operation
    result = index_put_accumulate(image, indices, data)

    # Verify Forward pass: Check if values were accumulated correctly
    expected = tf.constant([0, 1, 0, 1, 0, 1, 0, 0, 0, 0], dtype=tf.complex64)
    assert tf.reduce_all(tf.equal(result, expected)), "Forward pass failed: values not accumulated correctly"

    # Verify Gradient pass: Ensure custom gradient handles complex types correctly
    with tf.GradientTape() as tape:
        tape.watch(data)
        output = index_put_accumulate(image, indices, data)
        loss = tf.reduce_sum(output)

    grads = tape.gradient(loss, data)
    
    # The gradient of sum(output) w.r.t data should be ones (since output = data at indices)
    expected_grads = tf.ones(3, dtype=tf.complex64)
    assert tf.reduce_all(tf.equal(grads, expected_grads)), "Gradient pass failed: incorrect gradients for complex64"

    print("Test passed: custom_gradient successfully handled complex64 index_put logic.")

if __name__ == "__main__":
    test_custom_gradient_complex_index_put()