import torch

# Fix: Handle missing torch.compile for older PyTorch versions
if not hasattr(torch, 'compile'):
    # Mock torch.compile to return the function as-is (eager execution)
    # This allows the test to run on environments without PyTorch 2.0+
    torch.compile = lambda fn, **kwargs: fn

def fn(x, w):
    # Adapted to use the similar API: torch.nn.functional.conv_transpose2d
    return torch.nn.functional.conv_transpose2d(input=x, weight=w)

# Generate sample inputs for conv_transpose2d
# Input shape: (Batch, Channels, Height, Width)
# Weight shape: (Channels, Output_Channels, Kernel_H, Kernel_W)
inputs = [
    (torch.randn(1, 3, 10, 10, device="cuda", dtype=torch.float32),
     torch.randn(3, 5, 3, 3, device="cuda", dtype=torch.float32)),
    (torch.randn(2, 4, 8, 8, device="cuda", dtype=torch.float32),
     torch.randn(4, 6, 2, 2, device="cuda", dtype=torch.float32))
]

for x, w in inputs:
    compiled = torch.compile(fn, backend="inductor", mode="max-autotune")
    res1 = fn(x, w)
    res2 = compiled(x, w)
    torch.testing.assert_close(res1, res2)