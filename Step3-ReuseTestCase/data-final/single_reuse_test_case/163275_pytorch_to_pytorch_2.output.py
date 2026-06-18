import torch

# Create a tensor suitable for view_as_complex (float type, last dim size 2)
# Using float32 as it is the standard input for view_as_complex
A = torch.rand((1024, 2), device="cuda", dtype=torch.float32)

@torch.compile
def to_complex(input):
    return torch.view_as_complex(input)

# Run the compiled function
result = to_complex(A)

# Verify the result
assert result.dtype == torch.complex64
assert result.shape == (1024,)