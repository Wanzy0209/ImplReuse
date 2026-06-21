import torch
import sys

# Reproduce the specific configuration from the bug report
# These flags are crucial for triggering the specific lowering behavior
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    # Using torch.index_select as the similar API
    # arg0: input tensor
    # arg1: index tensor
    # arg2: dummy tensor to match original signature
    
    # Original API: torch.nn.functional.group_norm(t0, 1, weight=t2, bias=t2)
    # Similar API: torch.index_select(t0, dim, index)
    # We select along dimension 0 using arg1 as the index
    t3 = torch.index_select(arg0, 0, arg1)
    
    # Keep the subsequent operations to maintain graph complexity similar to the original
    t4 = t3.max(dim=0).values
    t5 = torch.sigmoid(t4)
    output = t5
    return output

# Setup inputs similar to the original report to maintain context
# arg0: size=(3, 395, 202, 357), dtype=bfloat16, device=cuda
arg0 = torch.rand([3, 395, 202, 357], dtype=torch.bfloat16, device='cuda', requires_grad=True)

# arg1: index tensor. 
# Original was size (1,), dtype=int64. 
# For index_select on dim 0 (size 3), indices must be < 3.
arg1 = torch.randint(0, 3, [5], dtype=torch.int64, device='cuda')

# arg2: size=(395,), dtype=bfloat16, device=cuda
arg2 = torch.rand([395], dtype=torch.bfloat16, device='cuda', requires_grad=True)

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0, arg1, arg2)
    out_eager.sum().backward()
    print('Eager Success! ')

    # Compiled execution
    # fullgraph=True and dynamic=True are used to stress the symbolic shape handling
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2)
    out_compiled.sum().backward()
    print('Compile Success! ')

    # Compare outputs (forward)
    out_eager_sum = out_eager.sum()
    out_compiled_sum = out_compiled.sum()
    diff = (out_eager_sum - out_compiled_sum).abs().item()
    rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100
    print(f'Relative diff (sum): {rel_diff:.6f}%')
    
    if rel_diff > 5:
        print(f' Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', out_eager_sum.item())
        print('out_compiled_sum:', out_compiled_sum.item())
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)