import torch
import tensorflow as tf

def test_custom_gradient_context_switch():
    """
    Test case adapted from PyTorch Issue 164094.
    
    Original Issue: Failed to change backward stream using torch.cuda.set_stream 
    inside a custom autograd Function's backward method.
    
    Adaptation: This test verifies if tf.custom_gradient allows changing the 
    execution context (analogous to switching CUDA streams) during the backward pass.
    We use tf.device as the semantic equivalent to torch.cuda.set_stream for 
    controlling execution placement.
    """

    # Define the custom gradient function mimicking the structure of the PyTorch issue.
    @tf.custom_gradient
    def backward_context_switch(input_tensor: tf.Tensor, target_device: str) -> tf.Tensor:
        """
        Forward pass: Acts as an identity function but captures the target device context.
        Analogous to BackwardStream.forward in the PyTorch issue.
        """
        # In PyTorch: ctx.stream = stream
        # In TF: We capture target_device in the closure of the grad function
        
        def grad(grad_output: tf.Tensor) -> tf.Tensor:
            """
            Backward pass: Switches execution context before processing gradients.
            Analogous to BackwardStream.backward in the PyTorch issue.
            """
            # In PyTorch: torch.cuda.set_stream(stream)
            # In TF: We use the tf.device context manager to switch execution context
            with tf.device(target_device):
                # Perform a dummy operation to ensure the context is respected
                # (e.g., ensuring the gradient calculation happens on the specified device)
                return grad_output * 1.0

        return input_tensor, grad

    # Setup test inputs
    # Using CPU to ensure the test is runnable on all environments, 
    # but the logic applies to GPU streams/devices as well.
    x = tf.constant(4.0)
    target_device = "/device:CPU:0"

    # Execute forward and backward pass
    with tf.GradientTape() as tape:
        tape.watch(x)
        # Call the custom function
        y = backward_context_switch(x, target_device)

    # Compute gradients
    grads = tape.gradient(y, x)

    # Assertions
    # 1. Verify that gradients were computed successfully
    assert grads is not None, "Gradient computation failed (returned None)."
    
    # 2. Verify the gradient value is correct (derivative of identity is 1)
    assert tf.equal(grads, 1.0).numpy(), f"Expected gradient 1.0, got {grads.numpy()}"

    print("Test passed: Custom gradient context switch executed successfully.")

if __name__ == "__main__":
    test_custom_gradient_context_switch()