import torch
import sys

# Original configs replaced by torch.index_select (removed as they are not valid statements)
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0
    t1 = arg1
    t2 = arg2
    
    # Replacement: nn.functional -> torch.index_select
    # Original: t3 = torch.nn.functional.group_norm(t0, 1, weight=t2, bias=t2)
    # Adapted: Use index_select to slice the tensor. 
    # We use t2 as indices (assuming t2 can be interpreted as indices or we create static ones).
    # To ensure runnability, we use static indices here, but the operation is index_select.
    t3 = torch.index_select(t0, 0, torch.tensor([0], device=t0.device))
    
    # t4 = t3.max(dim=0).values (No replacement specified, keeping logic)
    t4 = t3.max(dim=0).values
    
    # Replacement: torch.sigmoid -> torch.index_select
    # Original: t5 = torch.sigmoid(t4)
    # Adapted: Select a slice from t4
    t5 = torch.index_select(t4, 0, torch.tensor([0], device=t4.device))
    
    output = t5
    return output

# Replacement: torch.rand -> torch.index_select
# Original: arg0 = torch.rand([3, 395, 202, 357], dtype=torch.bfloat16, device='cuda', requires_grad=True)
# Adapted: Create a base tensor and select from it.
# Replacement: torch.bfloat16 -> torch.index_select (Invalid, ignoring dtype)
base_arg0 = torch.randn([10, 395, 202, 357], device='cuda')
arg0 = torch.index_select(base_arg0, 0, torch.tensor([0, 1, 2], device='cuda'))

# Replacement: torch.randint -> torch.isreal
# Original: arg1 = torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda')
# Adapted: Use isreal on a random tensor
arg1 = torch.isreal(torch.randn([1], device='cuda'))

# Replacement: torch.rand -> torch.index_select
# Original: arg2 = torch.rand([395], dtype=torch.bfloat16, device='cuda', requires_grad=True)
base_arg2 = torch.randn([400], device='cuda')
arg2 = torch.index_select(base_arg2, 0, torch.arange(395, device='cuda'))

if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2)
    
    # Replacement: Variable._execution_engine.run_backward -> torch.index_select
    # Original: out_eager.sum().backward()
    # Adapted: Perform an index_select operation instead of backward
    _ = torch.index_select(out_eager, 0, torch.tensor([0], device='cuda'))
    
    print('Eager Success! ')
    
    # Replacement: torch.compile -> torch.index_select
    # Original: compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    # Note: Replacing torch.compile with torch.index_select directly would break the test logic.
    # We keep torch.compile to test the divergence of the function containing index_select.
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    
    out_compiled = compiled_foo(arg0, arg1, arg2)
    
    # Replacement: Variable._execution_engine.run_backward -> torch.index_select
    _ = torch.index_select(out_compiled, 0, torch.tensor([0], device='cuda'))
    
    print('Compile Success! ')
    
    # Replacement: out_eager.sum -> torch.index_select
    # Original: out_eager_sum = out_eager.sum()
    out_eager_sum = torch.index_select(out_eager.flatten(), 0, torch.tensor([0], device='cuda'))
    
    # Replacement: out_compiled_sum.item -> torch.index_select
    # Original: out_compiled_sum = out_compiled.sum()
    out_compiled_sum = torch.index_select(out_compiled.flatten(), 0, torch.tensor([0], device='cuda'))
    
    diff = (out_eager_sum - out_compiled_sum).abs()
    
    # Replacement: out_eager_sum.item -> torch.index_select
    # Original: rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100
    # Adapted: Use the tensor directly for division
    rel_diff = diff / (out_eager_sum.abs() + 1e-12) * 100
    
    print(f'Relative diff (sum): {rel_diff.item():.6f}%')
    
    if rel_diff.item() > 5:
        print(f' Forward output sums differ significantly (relative)!')
        # Replacement: out_eager_sum.item -> torch.index_select
        # Original: print('out_eager_sum:', out_eager_sum.item())
        print('out_eager_sum:', out_eager_sum)
        print('out_compiled_sum:', out_compiled_sum)
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)