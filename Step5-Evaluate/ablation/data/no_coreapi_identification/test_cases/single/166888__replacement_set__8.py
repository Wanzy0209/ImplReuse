import torch
import torch.nn.functional as F

# Replacement: torch.randn -> torch.nn.functional.max_pool3d
# To use max_pool3d, we need a 5D input (N, C, D, H, W).
# We create a dummy input to feed into the replacement API.
dummy_input = torch.randn(1, 10, 20, 30, 1, device='cuda')
x = F.max_pool3d(dummy_input, kernel_size=2)

# Replacement: torch.tensor -> torch.cummin
# To use cummin, we need a tensor input.
# We create a dummy input to feed into the replacement API.
dummy_val = torch.tensor([5.0], device='cuda')
max_val_tuple = torch.cummin(dummy_val, dim=0)
max_val = max_val_tuple[0]

# Similar API: torch.nn.functional.tanh
# Bug Logic: Use .item() on a float tensor arg inside torch.compile
def f(x, max_val):
    # We leverage the similar API (tanh) and preserve the logic of using .item() on a tensor argument.
    # Since tanh does not take a scalar argument like clamp, we use the scalar to scale the result.
    y = F.tanh(x) * max_val.item()
    return y

compiled_func = torch.compile(f, backend='inductor', fullgraph=True)

# Run the compiled function
result = compiled_func(x, max_val)

# Assertion to verify the test case runs successfully
assert result is not None
assert result.shape == x.shape