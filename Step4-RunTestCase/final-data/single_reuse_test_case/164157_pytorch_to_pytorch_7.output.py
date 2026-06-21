import torch
import sys

# Fix: Check if torch._dynamo exists before accessing its attributes
# This prevents AttributeError in environments where TorchDynamo is not available (e.g., PyTorch < 2.0)
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

if hasattr(torch, '_inductor'):
    torch._inductor.config.emulate_precision_casts = True

def foo(a, b):
    # The bug involved an incompatibility between pointer<fp16> and triton.language.float64.
    # We use torch.promote_types to determine the common dtype between float16 and float64.
    # This tests the type promotion logic that might be causing the divergence.
    promoted_dtype = torch.promote_types(a.dtype, b.dtype)
    
    # Perform an operation to ensure the dtype is used in the graph
    # We cast 'a' to the promoted type and add 'b'
    # This forces the compiler to handle the type promotion logic
    return a.to(promoted_dtype) + b

# Inputs matching the types mentioned in the error: pointer<fp16> and triton.language.float64
# Using float16 and float64
# Note: This requires CUDA. If CUDA is not available, this will raise a RuntimeError.
try:
    a = torch.randn([256, 88, 1], dtype=torch.float16, device='cuda', requires_grad=True)
    b = torch.randn([256, 88, 1], dtype=torch.float64, device='cuda', requires_grad=True)
except RuntimeError as e:
    print(f"CUDA not available or error creating tensors: {e}")
    sys.exit(1)

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(a, b)
    out_eager.sum().backward()
    print('Eager Success! ')

    # Compiled execution
    # Fix: Check if torch.compile is available before attempting to use it
    if hasattr(torch, 'compile'):
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(a, b)
        out_compiled.sum().backward()
        print('Compile Success! ')
    else:
        print('torch.compile is not available in this environment. Skipping compiled execution.')