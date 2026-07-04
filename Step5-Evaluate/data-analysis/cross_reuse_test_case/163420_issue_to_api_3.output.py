import torch

# Preserve the configuration from the original bug report
# Use a try-except block to handle environments where torch._dynamo might not be exposed
try:
    torch._dynamo.config.capture_scalar_outputs = True
except AttributeError:
    pass

def foo(arg0, arg1):
    # Original issue: t2 = t0.clone(); t2.fill_diagonal_(t1.item())
    # Adaptation: Use torch.logaddexp with the 0-d tensor (arg1) directly.
    # This tests if the compiler handles the 0-d tensor broadcasting
    # correctly in a math operation context, similar to the scalar input issue in fill_diagonal_.
    return torch.logaddexp(arg0, arg1)

# Reproduce the exact tensor shapes and dtypes from the original issue
arg0 = torch.empty([1, 1], dtype=torch.float32, device='cuda', requires_grad=True)
arg1 = torch.empty([], dtype=torch.float32, device='cuda', requires_grad=True)

if __name__ == '__main__':
    # Test Eager mode
    out_eager = foo(arg0, arg1)
    out_eager.sum().backward()
    print('Eager Success! ')

    # Test Compiled mode with the same flags that triggered the bug
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1)
    out_compiled.sum().backward()
    
    # Verify consistency
    assert torch.allclose(out_eager, out_compiled), "Outputs differ between eager and compiled"
    print('Compile Success! ')