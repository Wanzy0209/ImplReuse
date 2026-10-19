import torch
import torch.nn as nn

def test_einsum_gradient_stride():
    """
    Test case to verify that torch.einsum calculates the correct stride 
    for the gradient of the input, matching the stride of the activation.
    """
    B, H, W, C = 20, 2, 2, 128

    # Setup input and linear layer
    x = torch.randn(B, H, W, C, requires_grad=True)
    linear = nn.Linear(C, C, bias=False)

    # Forward pass
    values = linear(x)
    values_view = values.view(B, H * W, C)

    # Hook to capture gradient stride
    grad_stride = None
    def hook(grad):
        nonlocal grad_stride
        grad_stride = grad.stride()
        return grad

    values_view.register_hook(hook)

    # Operation using torch.einsum
    weights = torch.randn(B, H * W, C)
    result = torch.einsum("bhc,bhc->bc", weights, values_view)

    # Backward pass
    result.backward(torch.ones_like(result))

    # Verification
    # According to the bug report, the gradient stride differs from the activation stride.
    # This test asserts that they should be equal.
    print(f"Activation stride: {values_view.stride()}")
    print(f"Gradient stride:   {grad_stride}")

    assert values_view.stride() == grad_stride, \
        f"Gradient stride mismatch! Expected {values_view.stride()}, got {grad_stride}"

if __name__ == "__main__":
    test_einsum_gradient_stride()