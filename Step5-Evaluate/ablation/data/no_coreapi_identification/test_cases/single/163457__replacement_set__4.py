import torch
import sys

# Keeping configurations to ensure the environment matches the bug report context
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(3, 395, 202, 357), stride=(239370, 79790, 202, 1), dtype=bfloat16, device=cuda
    t1 = arg1 # size=(1,), stride=(1,), dtype=int64, device=cuda
    t2 = arg2 # size=(395,), stride=(1,), dtype=bfloat16, device=cuda
    
    # Replacement: torch.nn.functional.group_norm -> torch.renorm
    # Original: t3 = torch.nn.functional.group_norm(t0, 1, weight=t2, bias=t2)
    # Using torch.renorm to mimic normalization behavior
    t3 = torch.renorm(t0, p=2, dim=1, max_norm=1.0)
    
    t4 = t3.max(dim=0).values # size=(395, 202, 357), stride=(72114, 357, 1), dtype=bfloat16, device=cuda
    
    # Replacement: torch.sigmoid -> torch.renorm
    # Original: t5 = torch.sigmoid(t4)
    # Applying torch.renorm as the replacement operation
    t5 = torch.renorm(t4, p=2, dim=0, max_norm=1.0)
    
    output = t5  # output tensor
    return output

# Replacement: torch.rand -> torch.renorm
# We generate data using randn and then apply torch.renorm to satisfy the replacement requirement
arg0 = torch.randn([3, 395, 202, 357], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg0 = torch.renorm(arg0, p=2, dim=1, max_norm=1.0)

# Replacement: torch.randint -> torch.fmax
# We generate data and apply torch.fmax
arg1 = torch.randn([1], dtype=torch.int64, device='cuda')
arg1 = torch.fmax(arg1, torch.tensor([0], device='cuda'))

# Replacement: torch.rand -> torch.renorm
arg2 = torch.randn([395], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg2 = torch.renorm(arg2, p=2, dim=0, max_norm=1.0)

if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2)
    out_eager.sum().backward()
    print('Eager Success! ')
    
    # Note: torch.compile is kept to test for divergence, as replacing it with torch.renorm would break the logic
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2)
    out_compiled.sum().backward()
    print('Compile Success! ')
    
    # Comparison logic (kept standard to ensure validity)
    # Replacements like out_eager.sum -> torch.renorm were ignored here to maintain test logic
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