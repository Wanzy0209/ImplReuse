import torch

# Define a function using the similar API: torch.all
def all_fn(x):
    return torch.all(x)

# Compile with the same settings as the bug report (fullgraph=True, backend="inductor")
inductor = torch.compile(all_fn, fullgraph=True, backend="inductor")

with torch.device("cuda"):
    # Use similar tensor shapes and dtypes as the original bug report
    # Original: q = torch.randn([2, 32, 4096, 128], dtype=torch.bfloat16, requires_grad=True)
    x = torch.randn([2, 32, 4096, 128], dtype=torch.bfloat16, requires_grad=True)

# Run forward pass
y = inductor(x)

# Note: torch.all returns a boolean scalar, so backward() is not applicable.
# This test verifies that torch.all compiles and runs correctly with the inductor backend.
print(f"Result: {y}")