import torch
import sys

# Check if torch._dynamo is available (requires PyTorch 2.0+)
if not hasattr(torch, '_dynamo'):
    print("Skipping test: torch._dynamo is not available. This test requires PyTorch 2.0+.")
    sys.exit(0)

# Reproduce the configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(input1, input2):
    # Replacing torch.var with torch.dist
    # torch.dist(input, other, p=2) computes the p-norm of (input - other)
    return torch.dist(input1, input2)

# Check for CUDA availability
if not torch.cuda.is_available():
    print("Skipping test: CUDA is not available.")
    sys.exit(0)

# Setup inputs with bfloat16 on CUDA, similar to the original bug context
input1 = torch.rand([10, 10], dtype=torch.bfloat16, device='cuda', requires_grad=True)
input2 = torch.rand([10, 10], dtype=torch.bfloat16, device='cuda', requires_grad=True)

if __name__ == '__main__':
    # Test Eager execution
    out_eager = foo(input1, input2)
    out_eager.backward()
    print('Eager Success! ')

    # Test Compiled execution
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(input1, input2)
    out_compiled.backward()
    print('Compile Success! ')