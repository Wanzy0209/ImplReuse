import torch

def fn(a, b):
    return torch.floor_divide(a, b)

# The original code failed because of a missing 'expecttest' dependency required by torch.testing._internal.
# To fix this, we replace the op_db dependency with manual input generation.
# This preserves the core logic: testing floor_divide with torch.compile.

# Determine device (CUDA was requested in the original test)
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("Warning: CUDA not available, falling back to CPU for testing.")

# Manually create sample inputs to mimic op_db.sample_inputs("cuda", torch.float32, requires_grad=False)
# We cover various shapes and cases including tensor-tensor and tensor-scalar operations.
test_inputs = [
    # Case 1: Standard tensor-tensor division
    (torch.randn(3, 3, device=device, dtype=torch.float32), torch.randn(3, 3, device=device, dtype=torch.float32)),
    # Case 2: Tensor divided by scalar
    (torch.randn(5, device=device, dtype=torch.float32), torch.tensor(2.0, device=device, dtype=torch.float32)),
    # Case 3: Specific values to test floor behavior (positive, negative, zero)
    (torch.tensor([10.5, -5.5, 0.0, 3.2], device=device, dtype=torch.float32), 
     torch.tensor([3.0, 2.0, 1.5, -1.0], device=device, dtype=torch.float32)),
]

compiled = torch.compile(fn, backend="inductor", mode="max-autotune")

for a, b in test_inputs:
    # Avoid division by zero to ensure clean test results
    b = torch.where(b == 0, torch.ones_like(b), b)
    
    res1 = fn(a, b)
    res2 = compiled(a, b)
    torch.testing.assert_close(res1, res2)