import tensorflow as tf
import torch
import sys

# Original configurations (kept as they are flags, replacing them with index_select is invalid syntax)
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, arg3, arg4):
    t0 = arg0
    
    # Replacement: torch.addmm -> torch.index_select
    # Original: t3 = torch.addmm(t0, t1, t2)
    # We select indices from t1 to simulate a transformation
    idx = torch.arange(4, device='cuda')
    t3 = torch.index_select(t1, 0, idx)
    
    # Replacement: t3.norm -> torch.index_select
    # Original: t4 = t3.norm()
    t4 = torch.index_select(t3, 1, torch.tensor([0], device='cuda'))
    
    t5 = arg3
    
    # Similar API: torch.quantile
    # Original: t6 = t5.var(dim=0)
    # Using torch.quantile to replace var logic
    q = torch.tensor([0.5], device='cuda')
    t6 = torch.quantile(t5, q, dim=0)
    
    # Original: t7 = t6.var()
    # Replacement: t7.view -> torch.index_select (No view in original, but replacing var logic)
    # We flatten to select a scalar-like value
    t7_flat = t6.flatten()
    t7 = torch.index_select(t7_flat, 0, torch.tensor([0], device='cuda'))
    
    t8 = arg4
    
    # Replacement: torch.nn.functional.relu -> torch.nn.functional.tanhshrink
    t9 = torch.nn.functional.tanhshrink(t8)
    
    t10 = t7 + t4 + t9
    
    # Replacement: torch.pow -> tf.keras.random.dropout
    # Using torch.nn.functional.dropout as the PyTorch equivalent
    # Original: t11 = torch.pow(torch.pow(t4, t7), t10)
    # We apply dropout to t4
    t11 = torch.nn.functional.dropout(t4, p=0.5)
    
    output = t11
    return output

# Input generation
# Replacement: torch.rand -> torch.index_select
# We create larger base tensors and select from them
base0 = torch.randn([10, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg0 = torch.index_select(base0, 0, torch.arange(5, device='cuda'))

base1 = torch.randn([10, 1024], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg1 = torch.index_select(base1, 0, torch.arange(5, device='cuda'))

base2 = torch.randn([2048, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg2 = torch.index_select(base2, 0, torch.arange(1024, device='cuda'))

base3 = torch.randn([10, 4, 5, 2], dtype=torch.float32, device='cuda', requires_grad=True)
arg3 = torch.index_select(base3, 0, torch.arange(3, device='cuda'))

base4 = torch.randn([5], dtype=torch.float32, device='cuda', requires_grad=True)
arg4 = torch.index_select(base4, 0, torch.tensor([0], device='cuda'))

if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2, arg3, arg4)
    out_eager.sum().backward()
    print('Eager Success! ')
    
    # torch.compile is kept to test the divergence
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4)
    out_compiled.sum().backward()
    print('Compile Success! ')
    
    # Compare outputs (forward)
    out_eager_sum = out_eager.sum()
    out_compiled_sum = out_compiled.sum()
    
    # Replacement: out_compiled_sum.item -> torch.index_select
    # We wrap the scalar difference in a tensor to apply index_select
    diff_tensor = (out_eager_sum - out_compiled_sum).abs().unsqueeze(0)
    diff_val = torch.index_select(diff_tensor, 0, torch.tensor([0], device='cuda'))
    
    # Replacement: out_eager_sum.item -> torch.index_select
    denom_tensor = (out_eager_sum.abs() + 1e-12).unsqueeze(0)
    denom_val = torch.index_select(denom_tensor, 0, torch.tensor([0], device='cuda'))
    
    rel_diff = (diff_val / denom_val) * 100
    
    print(f'Relative diff (sum): {rel_diff.item():.6f}%')
    
    if rel_diff.item() > 5:
        print(f' Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', out_eager_sum.item())
        print('out_compiled_sum:', out_compiled_sum.item())
        print('Absolute diff:', diff_val.item())
        print('Relative diff (%):', rel_diff.item())
        sys.exit(1)