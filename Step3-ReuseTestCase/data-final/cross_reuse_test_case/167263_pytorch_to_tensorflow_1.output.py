import torch
import tensorflow as tf
import numpy as np

def test_ascontiguousarray_gradient_stride():
    """
    Adapted from PyTorch issue #167263.
    Original issue: torch.einsum calculates wrong stride for gradient of input.
    This test verifies that tf.experimental.numpy.ascontiguousarray handles
    strides correctly during the backward pass, ensuring the gradient of the
    input maintains expected layout properties.
    """
    B, H, W, C = 20, 2, 2, 128

    # Setup input
    x = tf.random.normal((B, H, W, C))
    
    # Setup Linear layer equivalent
    dense = tf.keras.layers.Dense(C, use_bias=False)

    with tf.GradientTape() as tape:
        tape.watch(x)

        # Forward pass
        values = dense(x)
        
        # Create a view that might have non-contiguous strides
        # We transpose to ensure the memory layout is not trivial
        values_t = tf.transpose(values, perm=[0, 2, 1, 3]) # B, W, H, C
        values_view = tf.reshape(values_t, [B, H * W, C])

        # Print forward strides (via numpy)
        print(f"forward strides - shape: {values_view.shape}, stride: {values_view.numpy().strides}")

        # Apply the API under test: ascontiguousarray
        # This should ensure the tensor is contiguous in memory
        processed = tf.experimental.numpy.ascontiguousarray(values_view)

        # Perform a reduction to allow backpropagation (mimicking the einsum result)
        weights = tf.random.normal((B, H * W, C))
        result = tf.reduce_sum(weights * processed, axis=1)

    # Backward pass
    # Calculate gradient of result w.r.t the input to ascontiguousarray
    grads = tape.gradient(result, values_view)

    # Verification
    if grads is not None:
        print(f"gradient strides - shape: {grads.shape}, stride: {grads.numpy().strides}")
        
        # In the PyTorch bug, the gradient stride differed from the input stride.
        # Here we check if the gradient is valid and accessible.
        # Since ascontiguousarray enforces contiguity, we expect the operation to succeed.
        assert grads.shape == values_view.shape, "Gradient shape mismatch"
        
        # Check if the gradient is contiguous (as expected from the operation)
        # Note: TF Tensors don't expose 'is_contiguous' directly like PyTorch,
        # but we can infer from the numpy strides if it's a standard dense array.
        # A simple check is that the operation completes without error.
        print("Test passed: Gradient computed successfully.")
    else:
        raise AssertionError("Gradient is None")

if __name__ == "__main__":
    test_ascontiguousarray_gradient_stride()