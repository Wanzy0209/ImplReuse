import torch
import sys

# Configuration from the original test case
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    # Original logic used group_norm followed by max(dim=0).
    # We replace this with the similar API: torch.quantile.
    # torch.quantile reduces the dimension, similar to the effect of max(dim=0) in the original flow.
    # We apply quantile along dim=0 to match the output shape of the original t4.
    
    t0 = arg0 # size=(3, 395, 202, 357), dtype=bfloat16, device=cuda
    # arg1 is used as the quantile value (q). 
    # Note: torch.quantile expects q to be a scalar or tensor of floats in [0, 1].
    # We assume arg1 provided is suitable or adapt it.
    
    # Using torch.quantile (Similar API)
    # This replaces the group_norm + max(dim=0) sequence
    t3 = torch.quantile(t0, arg1, dim=0) 
    
    t4 = torch.sigmoid(t3)
    output = t4
    return output

# Input generation
# arg0: bfloat16 tensor, requires_grad (to test backward pass)
arg0 = torch.rand([3, 395, 202, 357], dtype=torch.bfloat16, device='cuda', requires_grad=True)

# arg1: Quantile value. Original was int64, but quantile needs float.
# We create a float tensor on cuda.
arg1 = torch.tensor([0.5], dtype=torch.float32, device='cuda')

# arg2: Unused in the modified logic but kept for signature consistency
arg2 = torch.rand([395], dtype=torch.bfloat16, device='cuda', requires_grad=True)

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0, arg1, arg2)
    out_eager.sum().backward()
    print('Eager Success! ')
    
    # Compiled execution
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2)
    out_compiled.sum().backward()
    print('Compile Success! ')
    
    # Comparison logic
    out_eager_sum = out_eager.sum()
    out_compiled_sum = out_compiled.sum()
    diff = (out_eager_sum - out_compiled_sum).abs().item()
    
    # Handle potential division by zero or very small values
    denominator = out_eager_sum.abs().item() + 1e-12
    rel_diff = diff / denominator * 100
    
    print(f'Relative diff (sum): {rel_diff:.6f}%')
    
    if rel_diff > 5:
        print(f' Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', out_eager_sum.item())
        print('out_compiled_sum:', out_compiled_sum.item())
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)