import torch
import tensorflow as tf
import numpy as np

def test_tf_tril_gradient_behavior():
    """
    Adapted test case for tf.experimental.numpy.tril based on torch.min gradient issue.
    
    Original Issue Logic:
    - torch.min (global) distributes gradients evenly among equal values.
    - torch.min (dim) assigns gradient to specific indices (sparse).
    
    Adaptation for tf.experimental.numpy.tril:
    - tril is a masking operation. We verify that gradients flow correctly
      through the mask (1.0 in lower triangle, 0.0 in upper triangle).
    """
    
    # Setup: Create a matrix of ones (similar to torch.ones)
    # Note: tf.experimental.numpy.tril requires rank >= 2
    x = tf.ones((4, 4), dtype=tf.float32)

    # Case 1: Standard tril (k=0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        # Apply operation
        y = tf.experimental.numpy.tril(x)
        # Reduce to scalar for backward pass
        loss = tf.reduce_sum(y)

    grads = tape.gradient(loss, x)

    # Expected: Gradients should be 1.0 in lower triangle, 0.0 elsewhere
    # This verifies the "indexing-like" or "masking" gradient behavior
    expected_grads = np.array([
        [1., 0., 0., 0.],
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.]
    ])

    print("Input:\n", x.numpy())
    print("Output (tril):\n", y.numpy())
    print("Gradients:\n", grads.numpy())
    
    assert np.allclose(grads.numpy(), expected_grads), \
        f"Gradient mismatch for tril. Expected:\n{expected_grads}\nGot:\n{grads.numpy()}"

    # Case 2: tril with offset k=1 (shifting the diagonal)
    # Verifies that gradient behavior adapts to the mask parameter
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_k = tf.experimental.numpy.tril(x, k=1)
        loss_k = tf.reduce_sum(y_k)

    grads_k = tape.gradient(loss_k, x)

    expected_grads_k = np.array([
        [1., 1., 0., 0.],
        [1., 1., 1., 0.],
        [1., 1., 1., 1.],
        [1., 1., 1., 1.]
    ])

    print("\nGradients (k=1):\n", grads_k.numpy())

    assert np.allclose(grads_k.numpy(), expected_grads_k), \
        f"Gradient mismatch for tril(k=1). Expected:\n{expected_grads_k}\nGot:\n{grads_k.numpy()}"

if __name__ == "__main__":
    test_tf_tril_gradient_behavior()