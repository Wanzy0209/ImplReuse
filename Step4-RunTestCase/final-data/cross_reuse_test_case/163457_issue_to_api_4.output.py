import torch
import sys

# Reproduce the specific configuration that triggers the bug
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    # Original code used group_norm. We replace it with the similar API: logaddexp.
    # arg0: size=(3, 395, 202, 357), dtype=bfloat16
    # arg2: size=(395,), dtype=bfloat16
    # Broadcasting arg2 to arg0's shape to test dynamic shape handling.
    t3 = torch.logaddexp(arg0, arg2)
    
    # Keep subsequent operations to ensure graph complexity matches the trigger scenario
    t4 = t3.max(dim=0).values
    t5 = torch.sigmoid(t4)
    output = t5
    return output

# Initialize inputs matching the original bug report's characteristics
arg0 = torch.rand([3, 395, 202, 357], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg1 = torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda')
arg2 = torch.rand([395], dtype=torch.bfloat16, device='cuda', requires_grad=True)

if __name__ == '__main__':
    # Run Eager mode
    out_eager = foo(arg0, arg1, arg2)
    out_eager.sum().backward()
    print('Eager Success! ')

    # Run Compiled mode with dynamic shapes
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2)
    out_compiled.sum().backward()
    print('Compile Success! ')

    # Validate results
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