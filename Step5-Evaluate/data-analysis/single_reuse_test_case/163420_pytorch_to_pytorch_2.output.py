import torch
import sys

# Check if torch._dynamo is available (requires PyTorch 2.0+)
if not hasattr(torch, '_dynamo'):
    print("Skipping test: torch._dynamo is not available (requires PyTorch 2.0+)")
    sys.exit(0)

# Check for CUDA availability as the test uses 'cuda' device
if not torch.cuda.is_available():
    print("Skipping test: CUDA is not available")
    sys.exit(0)

# Configuration from the original bug report to handle scalar outputs
torch._dynamo.config.capture_scalar_outputs = True

def foo(arg0, arg1):
    # Original: t2 = t0.clone(); t2.fill_diagonal_(t1.item())
    # Adaptation: Use torch.prod.
    # We use keepdim=True to maintain the (1, 1) shape of the original t2.
    # We multiply by arg1 to utilize the second argument, similar to the original logic.
    return torch.prod(arg0, keepdim=True) * arg1

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
    print('Compile Success! ')

    # Verify results match
    assert torch.allclose(out_eager, out_compiled)