import torch
import torch.library

# Define a custom operator that mimics a backend-specific operation
torch.library.define("my_custom_backend::custom_op", "(Tensor x) -> Tensor")

# Implement the forward logic for the custom operator
def custom_op_impl(x):
    # Example operation: multiply by 2
    return x * 2.0

torch.library.impl("my_custom_backend::custom_op", custom_op_impl)

# Define the backward function for the custom operator
def custom_op_backward(ctx, grad_output):
    # The derivative of x * 2 is 2
    return grad_output * 2.0

# Register the autograd function using the similar API
torch.library.register_autograd("my_custom_backend::custom_op", custom_op_backward)

# Test the setup
if __name__ == "__main__":
    # Create input data
    batch_size = 20
    input_data = torch.randn(batch_size, 10, requires_grad=True)

    # Execute the custom operator (Forward pass)
    output = torch.ops.my_custom_backend.custom_op(input_data)

    # Perform backward pass to verify autograd registration
    loss = output.sum()
    loss.backward()

    # Verify gradients
    expected_grad = torch.ones_like(input_data) * 2.0
    assert torch.allclose(input_data.grad, expected_grad), "Gradient calculation failed"
    
    print("success")