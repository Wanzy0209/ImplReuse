import torch

def forward(x):
    # Using torch.prod as the similar API
    # This replaces the repr() call from the original bug report
    return x * torch.prod(x)

x = torch.randn(2, 2)

# Compile with fullgraph=True, similar to the original bug report
compiled = torch.compile(forward, fullgraph=True)

# Execute the compiled function
result = compiled(x)

# Verify the result matches the eager execution
expected = forward(x)
assert torch.allclose(result, expected), "torch.compile failed to trace torch.prod correctly"