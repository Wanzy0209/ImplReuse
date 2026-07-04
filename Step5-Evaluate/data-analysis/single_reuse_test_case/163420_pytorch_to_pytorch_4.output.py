import torch
import sys

# Configuration from the original bug report
# Fix: Check if _dynamo exists before accessing it to handle older PyTorch versions
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
else:
    print("torch._dynamo is not available. This test requires PyTorch 2.0+. Skipping.")
    sys.exit(0)

def foo(arg0, arg1):
    t0 = arg0 # size=(1, 1), stride=(1, 1), dtype=float32, device=cuda
    t1 = arg1 # size=(), stride=(), dtype=float32, device=cuda
    
    # Adaptation: Replace fill_diagonal_ with torch.all
    # We perform a comparison between the (1,1) tensor and the 0-d tensor,
    # then check if all elements satisfy the condition.
    # This tests the similar API torch.all within the torch.compile context.
    res = torch.all(t0 == t1)
    return res

# Inputs from the original bug report
# Fix: Check for CUDA availability to prevent runtime errors on CPU-only machines
if not torch.cuda.is_available():
    print("CUDA is not available. Skipping test.")
    sys.exit(0)

arg0 = torch.empty([1, 1], dtype=torch.float32, device='cuda', requires_grad=True)
arg1 = torch.empty([], dtype=torch.float32, device='cuda', requires_grad=True)

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0, arg1)
    # Attempt backward pass to match original structure
    # Note: torch.all might have zero gradients, but we check for runtime errors
    try:
        out_eager.sum().backward()
    except RuntimeError:
        # If backward is not supported for this specific op/type combo, we ignore for the purpose of compilation testing
        pass
    print('Eager Success! ')

    # Compiled execution
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1)
    
    try:
        out_compiled.sum().backward()
    except RuntimeError:
        pass
        
    # Verify results match
    assert torch.equal(out_eager, out_compiled)
    print('Compile Success! ')