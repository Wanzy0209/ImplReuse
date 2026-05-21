import torch

# Setup input tensor (2D for transpose)
A = torch.rand((1024, 1024), device="cuda", dtype=torch.float16)

@torch.compile
def transpose_func(input):
    # torch.t does not support out_dtype, so we test the standard operation
    return torch.t(input)

# Execute the compiled function
result = transpose_func(A)

# Verify the result against eager execution
expected = torch.t(A)
assert torch.allclose(result, expected)