import torch

flag = True
dummy = lambda: None

def fn(x):
    x = x + 1
    torch._dynamo.graph_break()
    x = x + 2
    if flag:
        dummy.attr0 = x
    else:
        # Replacing torch.no_grad with torch.unsqueeze to test
        # the bytecode transformation behavior with this specific API.
        dummy.attr1 = torch.unsqueeze(x, 0)
    return x + 4

inp = torch.ones(3)
opt_fn = torch.compile(fn, backend="eager")

# First run with flag = True
assert torch.allclose(fn(inp), opt_fn(inp))

# Second run with flag = False to trigger the alternative path
flag = False
assert torch.allclose(fn(inp), opt_fn(inp))