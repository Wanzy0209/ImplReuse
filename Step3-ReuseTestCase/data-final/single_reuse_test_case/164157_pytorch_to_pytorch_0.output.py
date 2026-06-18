import torch
import sys

# Reproduce the specific configuration that triggers the bug
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(x):
    # Adapted to use torch.std (functional API) instead of tensor method
    # The bug involves type handling (fp16 vs float64) during compilation
    return torch.std(x, dim=2)

# Create a float16 tensor on CUDA, matching the context of the bug report
# Shape (256, 88, 4) is derived from the concatenation in the original issue
arg = torch.randn([256, 88, 4], dtype=torch.float16, device='cuda')

if __name__ == '__main__':
    # Test Eager mode
    try:
        out_eager = foo(arg)
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed: {e}")
        sys.exit(1)

    # Test Compiled mode
    try:
        compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
        out_compiled = compiled_foo(arg)
        print('Compile Success! ')
        
        # Verify consistency
        if not torch.allclose(out_eager, out_compiled):
            print("Divergence detected between eager and compiled outputs!")
            sys.exit(1)
    except Exception as e:
        print(f'Compile Failed: {e}')
        sys.exit(1)