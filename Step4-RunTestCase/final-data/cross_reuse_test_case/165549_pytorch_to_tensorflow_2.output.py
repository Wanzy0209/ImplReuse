import tensorflow as tf

# Enable eager execution to support .numpy() method on Tensors.
# The error 'Tensor' object has no attribute 'numpy' typically occurs
# when the code runs in graph mode (TF1 behavior) instead of eager mode (TF2).
if not tf.executing_eagerly():
    tf.compat.v1.enable_eager_execution()

def test_colocate_with_shape_preservation():
    """
    Adapted from PyTorch Issue 165549.
    
    Original Bug: Operations like abs dispatched via CompositeExplicitAutograd 
    return empty tensors (shape [0]) when run through cpu_fallback on a custom device.
    
    Target API: tf.compat.v1.colocate_with
    Adaptation: Verify that operations performed within the colocate_with context
    preserve the input tensor's shape and do not result in empty tensors.
    """
    # Create a tensor with a specific shape (4, 4)
    # In the original bug, this was on a 'privateuse1' device.
    # Here we use the default device, but enforce colocation.
    t = tf.random.normal((4, 4))
    
    # Use the similar API to ensure operations run on the same device as 't'
    with tf.compat.v1.colocate_with(t):
        # Perform the operation mentioned in the original bug report
        result = tf.abs(t)
    
    # The original bug resulted in shape [0]. 
    # We assert that the shape is preserved correctly.
    assert result.shape == t.shape, (
        f"Shape mismatch: expected {t.shape}, got {result.shape}. "
        "This mirrors the PyTorch bug where abs returned an empty tensor."
    )
    
    # Explicitly check that the tensor is not empty
    assert tf.size(result).numpy() != 0, "Result tensor is empty (size 0)"
    
    print("Test passed: Shape preserved correctly.")

if __name__ == "__main__":
    test_colocate_with_shape_preservation()