import torch
import torch.nn.functional as F

# Generate input data using torch.nn.functional.pdist instead of torch.tensor
# We need a (3, 3) matrix 'a' and a (3, 1) vector 'x', totaling 10 elements.
# pdist on a (5, 2) tensor returns 5*4/2 = 10 elements.
base_input = torch.randn(5, 2)
pdist_output = F.pdist(base_input)

a = pdist_output[:9].reshape(3, 3)
x = pdist_output[9:].reshape(3, 1)

def f(a, x):
    # Replaced torch.sparse.mm with torch.mm
    return torch.mm(a, x)

# Test direct execution
result = f(a, x)
assert result.shape == (3, 1), "Direct call shape mismatch"

# Test with vjp (the context of the bug)
# The bug report indicates vjp fails for sparse.mm due to layout loss.
# We verify torch.mm works correctly with vjp.
try:
    vjp_fn = torch.func.vjp(f, a, x)[1]
    # Compute gradients to ensure it runs end-to-end
    grads = vjp_fn(torch.ones_like(result))
    assert grads[0].shape == a.shape, "Gradient shape mismatch for a"
    assert grads[1].shape == x.shape, "Gradient shape mismatch for x"
    print("Test passed: torch.mm works with vjp.")
except Exception as e:
    print(f"Test failed: {e}")
    raise