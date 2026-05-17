import torch
import tensorflow as tf
import numpy as np

def test_tril_gradient_behavior():
    """
    Adapted test case for tf.keras.ops.tril based on the torch.min gradient issue.
    
    The original issue highlights that torch.min distributes gradients evenly when 
    reducing globally, but uses indexing (gradient 1.0 for selected, 0.0 for others) 
    when reducing over a dimension.
    
    tf.keras.ops.tril is a masking/indexing operation. We verify that its gradient 
    behavior matches the "indexing" pattern (gradients flow only to the selected 
    lower-triangular elements) rather than distributing gradients arbitrarily 
    across equal values.
    """
    
    # Create a matrix of ones to ensure all input values are equal
    # This mirrors the setup in the original torch.min bug report
    a = tf.ones([3, 3])

    with tf.GradientTape() as tape:
        tape.watch(a)
        # Apply tril operation
        # This selects the lower triangle, effectively an indexing/masking operation
        result = tf.keras.ops.tril(a)

    # To compute gradients, we need a scalar loss. 
    # We sum the result, which accumulates the gradients of the selected elements.
    # This is analogous to calling backward() on the output tensor.
    loss = tf.reduce_sum(result)
    grads = tape.gradient(loss, a)

    # Expected behavior:
    # Gradients should be 1.0 for elements in the lower triangle (selected by tril).
    # Gradients should be 0.0 for elements in the upper triangle (masked out).
    # This confirms the "indexing" behavior similar to torch.min(dim=...).
    expected_grads = np.array([
        [1., 0., 0.],
        [1., 1., 0.],
        [1., 1., 1.]
    ])

    print("Input (ones):\n", a.numpy())
    print("Output (tril):\n", result.numpy())
    print("Gradients:\n", grads.numpy())

    # Assert that gradients match the expected mask pattern
    # This verifies that gradients are not distributed evenly among all equal values,
    # but are assigned specifically to the indices selected by the operation.
    assert np.allclose(grads.numpy(), expected_grads), \
        f"Gradient mismatch.\nExpected:\n{expected_grads}\nGot:\n{grads.numpy()}"

if __name__ == "__main__":
    test_tril_gradient_behavior()