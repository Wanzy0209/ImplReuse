import torch
import tensorflow as tf
import numpy as np

def test_broadcast_arrays_gradient():
    """
    Adapted test case to verify gradient behavior of tf.experimental.numpy.broadcast_arrays.
    
    The original issue highlighted inconsistent gradient distribution in torch.min 
    (even distribution vs indexing). For broadcast_arrays, the expected behavior is 
    that gradients are summed back to the original shape where broadcasting occurred.
    This test verifies that gradients flow correctly through the broadcasting operation.
    """
    
    print("--- Test Case 1: Broadcasting scalar to vector ---")
    # Input: a vector and a scalar
    a = tf.constant([1.0, 2.0, 3.0])
    b = tf.constant(1.0)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([a, b])
        # broadcast_arrays returns a list of arrays broadcasted to a common shape
        # a is [1, 2, 3], b is [1] -> broadcasted b becomes [1, 1, 1]
        res_a, res_b = tf.experimental.numpy.broadcast_arrays(a, b)
        
        # Compute a scalar loss (sum of all elements)
        # Loss = (1+2+3) + (1+1+1) = 9
        loss = tf.reduce_sum(res_a) + tf.reduce_sum(res_b)

    # Calculate gradients
    grad_a = tape.gradient(loss, a)
    grad_b = tape.gradient(loss, b)

    print(f"Input a: {a.numpy()}")
    print(f"Input b: {b.numpy()}")
    print(f"Gradient for a: {grad_a.numpy()}") # Expected: [1., 1., 1.]
    print(f"Gradient for b: {grad_b.numpy()}") # Expected: 3.0 (sum of gradients from the 3 broadcasted positions)
    
    # Assertions
    assert np.array_equal(grad_a.numpy(), [1., 1., 1.]), "Gradient for vector input should be ones."
    assert grad_b.numpy() == 3.0, "Gradient for scalar input should be sum of output gradients."

    print("\n--- Test Case 2: Broadcasting 1D to 2D (Dimensional expansion) ---")
    # Input: (2, 1) and (1, 2) -> broadcast to (2, 2)
    c = tf.constant([[1.0], [2.0]]) 
    d = tf.constant([[1.0, 2.0]])

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([c, d])
        res_c, res_d = tf.experimental.numpy.broadcast_arrays(c, d)
        
        # res_c becomes [[1, 1], [2, 2]]
        # res_d becomes [[1, 2], [1, 2]]
        loss = tf.reduce_sum(res_c) + tf.reduce_sum(res_d)

    grad_c = tape.gradient(loss, c)
    grad_d = tape.gradient(loss, d)

    print(f"Input c shape: {c.shape}, values:\n{c.numpy()}")
    print(f"Input d shape: {d.shape}, values:\n{d.numpy()}")
    print(f"Gradient for c:\n{grad_c.numpy()}") # Expected: [[2], [2]] (summed over columns)
    print(f"Gradient for d:\n{grad_d.numpy()}") # Expected: [[2, 2]] (summed over rows)

    # Assertions
    assert np.array_equal(grad_c.numpy(), [[2.], [2.]]), "Gradient for c should be summed over the broadcasted dimension."
    assert np.array_equal(grad_d.numpy(), [[2., 2.]]), "Gradient for d should be summed over the broadcasted dimension."

    print("\nAll gradient checks passed for tf.experimental.numpy.broadcast_arrays.")

if __name__ == "__main__":
    test_broadcast_arrays_gradient()