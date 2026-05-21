import torch
import torch.nn.functional as F

# Setup for the similar API: torch.nn.functional.conv_transpose3d
# Input needs to be 5D for conv_transpose3d: (Batch, Channel, Depth, Height, Width)
inp = torch.ones(1, 1, 3, 3, 3)
# Weight needs to match input/output channels and kernel size
weight = torch.randn(1, 1, 3, 3, 3)

flag = True
dummy = lambda: None

def fn(x):
    x = x + 1
    torch._dynamo.graph_break()
    x = x + 2
    if flag:
        dummy.attr0 = x
    else:
        with torch.no_grad():
            # Leverage the similar API: torch.nn.functional.conv_transpose3d
            # This replaces the simple assignment with a complex operation
            # to test bytecode transformation under the context manager.
            res = F.conv_transpose3d(x, weight)
            dummy.attr1 = res
    return x + 4

opt_fn = torch.compile(fn, backend="eager")

# First run with flag=True
assert torch.allclose(fn(inp), opt_fn(inp))

# Second run with flag=False to trigger the else block with no_grad and the similar API
flag = False
assert torch.allclose(fn(inp), opt_fn(inp))