import torch

# Handle environments where torch.compile is not available (PyTorch < 2.0)
if hasattr(torch, 'compile'):
    decorator = torch.compile(fullgraph=False, backend="eager")
else:
    # Pass-through decorator for older PyTorch versions
    def decorator(func):
        return func

@decorator
def func(a):
    # Adapted from u0, u1 = a.tolist()
    # torch.any returns a boolean scalar indicating if any element is true
    u0 = torch.any(a > 0)
    
    # Adapted from return a*u0*u1
    # Use the scalar result in the computation
    return a * u0

# Test case 1: Condition is True (elements > 0)
input_tensor = torch.tensor([1, 2])
output = func(input_tensor)
assert torch.equal(output, input_tensor)

# Test case 2: Condition is False (no elements > 0)
input_tensor = torch.tensor([-1, -2])
output = func(input_tensor)
assert torch.equal(output, torch.tensor([0, 0]))