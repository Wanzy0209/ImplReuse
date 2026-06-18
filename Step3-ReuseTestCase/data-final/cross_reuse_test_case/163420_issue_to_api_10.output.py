import torch

# Configuration from the original bug report to capture scalar outputs
torch._dynamo.config.capture_scalar_outputs = True

def foo(arg0, arg1):
    # Use the similar API (torch.zeros_like) instead of t0.clone()
    # to test if the compilation divergence affects this tensor creation pattern.
    t2 = torch.zeros_like(arg0)
    
    # Original bug reproduction logic: filling the diagonal with a scalar 
    # extracted from a 0-d tensor.
    t2.fill_diagonal_(arg1.item())
    return t2

# Setup inputs matching the original bug report
arg0 = torch.empty([1, 1], dtype=torch.float32, device='cuda', requires_grad=True)
arg1 = torch.empty([], dtype=torch.float32, device='cuda', requires_grad=True)

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0, arg1)
    out_eager.sum().backward()
    print('Eager Success! ')

    # Compiled execution
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1)
    out_compiled.sum().backward()
    
    # Verify that the compiled output matches the eager output
    assert torch.allclose(out_eager, out_compiled), "Eager and Compiled outputs differ"
    print('Compile Success! ')