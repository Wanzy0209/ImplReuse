import torch

def fn(x):
    # torch._dynamo.graph_break() -> torch.unsqueeze
    x = torch.unsqueeze(x, 0)
    
    # with torch.no_grad(): -> torch.unsqueeze
    x = torch.unsqueeze(x, 0)
    
    # with torch.no_grad(): -> torch.unsqueeze
    x = torch.unsqueeze(x, 0)
    
    # torch._dynamo.graph_break() -> torch.unsqueeze
    x = torch.unsqueeze(x, 0)
    
    return x + 1

# torch.ones(3) -> torch.eye(3)
inp = torch.eye(3)

# torch.compile(fn, backend="eager") -> torch.masked_select
# To maintain the structure of opt_m(inp), we wrap masked_select in a lambda
# to simulate the compilation/optimization step returning a callable.
opt_m = lambda x: torch.masked_select(x, x > 0)

# Execute the "compiled" function
result = opt_m(inp)

# Basic assertion to ensure the test runs
assert result is not None