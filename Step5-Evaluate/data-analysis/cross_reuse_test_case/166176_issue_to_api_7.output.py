import torch
import torch.nn.functional as F

def fn(x, weight):
    # First graph break
    torch._dynamo.graph_break()
    
    with torch.no_grad():
        # Nested no_grad context
        with torch.no_grad():
            # Second graph break inside the nested context
            # This pattern triggers the resume codegen KeyError in the original issue
            torch._dynamo.graph_break()
            
            # Leverage the similar API (conv_transpose3d) inside the problematic context
            x = F.conv_transpose3d(x, weight)
            
    return x

# Setup inputs suitable for conv_transpose3d
# Input shape: (Batch, Channels, Depth, Height, Width)
inp = torch.randn(1, 3, 5, 5, 5)
# Weight shape: (Input Channels, Output Channels, kD, kH, kW)
weight = torch.randn(3, 3, 3, 3, 3)

# Compile the function with the eager backend to trigger the dynamo logic
opt_m = torch.compile(fn, backend="eager")

# Run the compiled function
# This should reproduce the error if the bug exists, or pass if fixed
result = opt_m(inp, weight)

# Assertion to verify the operation executed correctly
# Output size calculation: (5 - 1)*1 + 2*(3-1) + 1 = 7
assert result.shape == (1, 3, 7, 7, 7)