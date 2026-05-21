import torch


def f(x):
    # view_as_complex has specific requirements on strides and dtype
    return torch.view_as_complex(x)


# Create a valid input for view_as_complex:
# Must be floating point, last dimension size 2, and contiguous (stride 1 on last dim)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Test direct execution (sanity check)
print("Direct execution result:", f(x))

# Test inside torch.func.vjp
# The original bug occurred because GradTrackingTensor did not preserve layout.
# view_as_complex is sensitive to strides, so this tests if metadata is preserved correctly.
try:
    # Construct the vjp
    vjp_fn = torch.func.vjp(f, x)[1]
    
    # Execute the vjp with a cotangent to ensure the backward pass also works
    # The output of f is complex, so the cotangent must be complex
    cotangent = torch.ones(3, dtype=torch.complex64)
    vjp_fn(cotangent)
    
    print("VJP test passed successfully.")
except Exception as e:
    print(f"VJP test failed with error: {e}")