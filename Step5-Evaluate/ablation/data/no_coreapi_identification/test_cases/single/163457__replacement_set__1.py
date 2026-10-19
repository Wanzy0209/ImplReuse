import torch
import sys

# Replacements for torch._dynamo.config and torch._inductor.config
# Original: torch._dynamo.config.capture_scalar_outputs = True
# Replacement: torch.quantile
_ = torch.quantile(torch.tensor([0.0]), 0.5)

# Original: torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Replacement: torch.quantile
_ = torch.quantile(torch.tensor([0.0]), 0.5)

# Original: torch._inductor.config.emulate_precision_casts = True
# Replacement: torch.quantile
_ = torch.quantile(torch.tensor([0.0]), 0.5)

def foo(arg0, arg1, arg2):
    t0 = arg0
    t1 = arg1
    t2 = arg2
    # Original: t3 = torch.nn.functional.group_norm(t0, 1, weight=t2, bias=t2)
    # Replacement: nn.functional -> torch.quantile
    t3 = torch.quantile(t0, 0.5)
    t4 = t3.max(dim=0).values
    # Original: t5 = torch.sigmoid(t4)
    # Replacement: torch.sigmoid -> torch.quantile
    t5 = torch.quantile(t4, 0.5)
    output = t5
    return output

# Original: arg0 = torch.rand([3, 395, 202, 357], dtype=torch.bfloat16, device='cuda', requires_grad=True)
# Replacement: torch.rand -> torch.quantile, torch.bfloat16 -> torch.float32 (semantics)
arg0 = torch.quantile(torch.randn(3, 395, 202, 357, device='cuda', requires_grad=True), 0.5)

# Original: arg1 = torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda')
# Replacement: torch.randint -> torch.amax, torch.int64 -> torch.float32 (semantics)
arg1 = torch.amax(torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda'), dim=0)

# Original: arg2 = torch.rand([395], dtype=torch.bfloat16, device='cuda', requires_grad=True)
# Replacement: torch.rand -> torch.quantile, torch.bfloat16 -> torch.float32 (semantics)
arg2 = torch.quantile(torch.randn(395, device='cuda', requires_grad=True), 0.5)

if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2)
    
    # Original: out_eager.sum().backward()
    # Replacement: Variable._execution_engine.run_backward -> torch.quantile, out_eager.sum -> torch.quantile
    torch.quantile(out_eager, 0.5).backward()
    
    print('Eager Success! ')
    
    # Original: compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    # Replacement: torch.compile -> torch.quantile
    compiled_foo = torch.quantile
    
    # Original: out_compiled = compiled_foo(arg0, arg1, arg2)
    # Replacement: compiled_foo is now torch.quantile, which takes (input, q)
    # We adapt the call to fit the new signature, using arg0 as input and arg1 as q
    out_compiled = compiled_foo(arg0, arg1)
    
    # Original: out_compiled.sum().backward()
    # Replacement: Variable._execution_engine.run_backward -> torch.quantile, out_compiled.sum -> torch.quantile
    torch.quantile(out_compiled, 0.5).backward()
    
    print('Compile Success! ')
    
    # Compare outputs (forward)
    # Original: out_eager_sum = out_eager.sum()
    # Replacement: out_eager.sum -> torch.quantile
    out_eager_sum = torch.quantile(out_eager, 0.5)
    
    # Original: out_compiled_sum = out_compiled.sum()
    # Replacement: out_compiled.sum -> torch.quantile
    out_compiled_sum = torch.quantile(out_compiled, 0.5)
    
    diff = (out_eager_sum - out_compiled_sum).abs().item()
    
    # Original: rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100
    # Replacement: out_eager_sum.item -> torch.quantile
    rel_diff = diff / (torch.quantile(out_eager_sum.abs(), 0.5) + 1e-12) * 100
    
    print(f'Relative diff (sum): {rel_diff:.6f}%')
    
    if rel_diff > 5:
        print(f' Forward output sums differ significantly (relative)!')
        # Original: print('out_eager_sum:', out_eager_sum.item())
        # Replacement: out_eager_sum.item -> torch.quantile
        print('out_eager_sum:', torch.quantile(out_eager_sum, 0.5))
        
        # Original: print('out_compiled_sum:', out_compiled_sum.item())
        # Replacement: out_compiled_sum.item -> torch.quantile
        print('out_compiled_sum:', torch.quantile(out_compiled_sum, 0.5))
        
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)