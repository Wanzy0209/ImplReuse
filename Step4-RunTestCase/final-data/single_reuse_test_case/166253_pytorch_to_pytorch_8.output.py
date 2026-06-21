import torch

def func_nojit(x):
    # Adapted from torch.full to torch.trace
    return torch.trace(x)

# Check if torch.compile is available (requires PyTorch 2.0+)
if hasattr(torch, 'compile'):
    func_jit = torch.compile(func_nojit)
    funcs = [func_nojit, func_jit]
else:
    print("Warning: torch.compile is not available in this PyTorch version. Skipping JIT test.")
    funcs = [func_nojit]

for func in funcs:
    # Inputs must be 2D tensors for torch.trace
    # Using float64 as per the original bug report
    x1 = torch.tensor([[5.0, 0.0], [0.0, 5.0]], dtype=torch.float64)
    x2 = torch.tensor([[10.0, 0.0], [0.0, 10.0]], dtype=torch.float64)
    
    res1 = func(x1)
    res2 = func(x2)
    
    print(res1)
    print(res2)
    
    # Assertions to verify the behavior matches expectations
    # and catches the potential caching bug
    assert res1.item() == 10.0, f"Expected 10.0, got {res1.item()}"
    assert res2.item() == 20.0, f"Expected 20.0, got {res2.item()}"