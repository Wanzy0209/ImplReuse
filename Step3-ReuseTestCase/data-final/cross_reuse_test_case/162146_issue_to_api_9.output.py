import torch

# Set seed for reproducibility
torch.manual_seed(2025)

def foo(x):
    # Replace sin_ with sinc. Since sinc is not in-place, we assign back to the slice.
    # This tests the index_put_ behavior with torch.sinc under torch.compile.
    x[0] = torch.sinc(x[0])
    x[1] = torch.sinc(x[1])
    y = torch.zeros_like(x)
    y[2] = x[0]
    y[3] = x[1]
    return y

# Compile the function
cfoo = torch.compile(foo)

# Create input tensors. 
# Using randn because sinc(n) is 0 for integers, which makes for a weak test.
x = torch.randn(4, 3, dtype=torch.float32)
cx = x.clone()

# Execute eager and compiled versions
res = foo(x)
cres = cfoo(cx)

# Assert that the results are close
try:
    torch.testing.assert_close(res, cres)
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")