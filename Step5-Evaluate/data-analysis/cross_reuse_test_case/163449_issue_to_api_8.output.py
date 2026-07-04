import torch
import sys

# Reproduce the configuration from the original bug report
# These settings are crucial for triggering the specific divergence path
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(input_tensor, index_tensor, var_tensor, scalar_input):
    # Original bug used torch.addmm here.
    # We replace it with the similar API: torch.index_select.
    # input_tensor: size=(5, 4), dtype=bfloat16, device=cuda
    # index_tensor: size=(3,), dtype=int64, device=cuda
    t3 = torch.index_select(input_tensor, 0, index_tensor) # size=(3, 4), dtype=bfloat16, device=cuda

    # Preserve the rest of the graph to test mixed precision and scalar handling
    # which was the context of the eager/compile divergence.
    t4 = t3.norm() # size=(), stride=(), dtype=bfloat16, device=cuda
    t5 = var_tensor # size=(3, 4, 5, 2), stride=(40, 10, 2, 1), dtype=float32, device=cuda
    t6 = t5.var(dim=0) # size=(4, 5, 2), stride=(10, 2, 1), dtype=float32, device=cuda
    t7 = t6.var() # size=(), stride=(), dtype=float32, device=cuda
    t8 = scalar_input # size=(), stride=(), dtype=float32, device=cuda
    t9 = torch.nn.functional.relu(t8) # size=(), stride=(), dtype=float32, device=cuda
    t10 = t7 + t4 + t9 # size=(), stride=(), dtype=float32, device=cuda
    t11 = torch.pow(torch.pow(t4, t7), t10) # size=(), stride=(), dtype=float32, device=cuda
    output = t11  # output tensor
    return output

# Setup inputs mimicking the original shapes and types
arg0 = torch.rand([5, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True)
# Index tensor for index_select (indices must be valid for dimension 0 of arg0)
arg1 = torch.tensor([0, 2, 4], device='cuda', dtype=torch.long)
arg2 = torch.rand([3, 4, 5, 2], dtype=torch.float32, device='cuda', requires_grad=True)
arg3 = torch.rand([], dtype=torch.float32, device='cuda', requires_grad=True)

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0, arg1, arg2, arg3)
    out_eager.sum().backward()
    print('Eager Success! ')

    # Compiled execution
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2, arg3)
    out_compiled.sum().backward()
    print('Compile Success! ')

    # Compare outputs (forward)
    out_eager_sum = out_eager.sum()
    out_compiled_sum = out_compiled.sum()
    diff = (out_eager_sum - out_compiled_sum).abs().item()
    rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100
    print(f'Relative diff (sum): {rel_diff:.6f}%')
    
    # Check for significant divergence similar to the original bug report
    if rel_diff > 5:
        print(f' Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', out_eager_sum.item())
        print('out_compiled_sum:', out_compiled_sum.item())
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)