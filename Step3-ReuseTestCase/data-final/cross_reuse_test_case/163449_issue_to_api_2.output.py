import torch
import sys

# Reproduce the specific configuration from the bug report that triggers the divergence
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1):
    # arg0: bfloat16 tensor, similar to the inputs in the original bug
    t0 = arg0 
    # Use torch.median (the similar API) instead of addmm/norm
    # This tests if median exhibits the same eager/compile divergence under mixed precision
    t1 = torch.median(t0) 

    # arg1: float32 tensor, similar to the secondary inputs in the original bug
    t2 = arg1 
    t3 = t2.var()

    # Combine the results to check for numerical divergence
    # This mimics the scalar arithmetic logic in the original bug (t7 + t4 + t9)
    output = t1 + t3
    return output

# Setup inputs
# Using bfloat16 to trigger the precision casting emulation logic suspected in the bug
arg0 = torch.randn([5, 1024], dtype=torch.bfloat16, device='cuda')
arg1 = torch.randn([3, 4, 5, 2], dtype=torch.float32, device='cuda')

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0, arg1)
    print('Eager Success! ')

    # Compiled execution
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1)
    print('Compile Success! ')

    # Compare outputs (forward)
    # Since the output is a scalar here, we compare directly
    diff = (out_eager - out_compiled).abs().item()
    rel_diff = diff / (out_eager.abs().item() + 1e-12) * 100
    
    print(f'Relative diff: {rel_diff:.6f}%')
    
    if rel_diff > 5:
        print(f' Forward outputs differ significantly (relative)!')
        print('out_eager:', out_eager.item())
        print('out_compiled:', out_compiled.item())
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)