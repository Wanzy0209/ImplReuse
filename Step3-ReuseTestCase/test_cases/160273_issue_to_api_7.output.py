import torch
import tensorflow as tf

def test_concatenate_gradient_behavior():
    """
    Test case to verify gradient flow through tf.keras.ops.concatenate.
    This mirrors the logic of the original torch.min issue by checking
    how gradients are distributed back to the input tensors after the operation.
    """
    
    # Setup input tensors
    # Using ones to easily verify gradient magnitude (should be 1.0 for sum reduction)
    a = tf.ones([2, 3])
    b = tf.ones([2, 3])

    # --- Test Case 1: Concatenate along axis 0 ---
    with tf.GradientTape() as tape:
        tape.watch([a, b])
        
        # Perform operation: concatenate along axis 0
        # Analogous to checking a specific mode of the operation
        result = tf.keras.ops.concatenate([a, b], axis=0)
        
        # Define a scalar loss to backpropagate (similar to min_val.backward())
        loss = tf.reduce_sum(result)

    # Calculate gradients
    grads = tape.gradient(loss, [a, b])

    # Verify gradients flow correctly to both inputs
    # Unlike torch.min (which might distribute evenly or sparsely),
    # concatenate should map gradients directly back to the source tensors.
    assert grads[0].shape == a.shape, "Gradient shape for 'a' does not match input"
    assert grads[1].shape == b.shape, "Gradient shape for 'b' does not match input"
    
    # Since we summed the output and inputs are ones, gradients should be all ones
    assert tf.reduce_all(grads[0] == 1.0), "Gradient values for 'a' are incorrect"
    assert tf.reduce_all(grads[1] == 1.0), "Gradient values for 'b' are incorrect"

    # --- Test Case 2: Concatenate along axis -1 (default) ---
    # Mirroring the pattern of testing different configurations (dim vs no-dim in torch.min)
    with tf.GradientTape() as tape:
        tape.watch([a, b])
        
        # Perform operation: concatenate along default axis (-1)
        result = tf.keras.ops.concatenate([a, b])
        loss = tf.reduce_sum(result)

    grads = tape.gradient(loss, [a, b])

    assert grads[0].shape == a.shape
    assert grads[1].shape == b.shape
    assert tf.reduce_all(grads[0] == 1.0)
    assert tf.reduce_all(grads[1] == 1.0)

    print("Test passed: Gradients distributed correctly for tf.keras.ops.concatenate.")

if __name__ == "__main__":
    test_concatenate_gradient_behavior()