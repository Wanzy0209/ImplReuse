import torch
import torch.nn as nn

def test_einsum_gradient_stride():
    """
    Test case to verify that torch.einsum calculates the correct stride 
    for the gradient of the input tensor.
    
    Based on Issue ID: 167263
    """
    B, H, W, C = 20, 2, 2, 128

    x = torch.randn(B, H, W, C, requires_grad=True)
    linear = nn.Linear(C, C, bias=False)

    values = linear(x)
    values_view = values.view(B, H * W, C)
    values_view.retain_grad()

    # Store expected stride
    expected_stride = values_view.stride()

    weights = torch.randn(B, H * W, C)
    
    # Call the API under test
    result = torch.einsum("bhc,bhc->bc", weights, values_view)

    result.backward(torch.ones_like(result))

    # Verify gradient exists
    assert values_view.grad is not None, "Gradient for values_view is None"

    # Verify stride matches the input tensor's stride
    actual_stride = values_view.grad.stride()
    
    print(f"Input shape: {values_view.shape}, Input stride: {expected_stride}")
    print(f"Grad  shape: {values_view.grad.shape}, Grad  stride: {actual_stride}")

    assert expected_stride == actual_stride, (
        f"Gradient stride mismatch!\n"
        f"Expected: {expected_stride}\n"
        f"Actual:   {actual_stride}"
    )

if __name__ == "__main__":
    test_einsum_gradient_stride()