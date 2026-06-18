import torch
import sys

# Configuration from the original bug report to trigger the specific behavior
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg3, arg4):
    # arg0: size=(5, 4), dtype=bfloat16, device=cuda
    # arg3: size=(3, 4, 5, 2), dtype=float32, device=cuda
    # arg4: size=(), dtype=float32, device=cuda

    t0 = arg0
    
    # Original code used: t3 = torch.addmm(t0, t1, t2)
    # We replace this with the similar API: torch.zeros_like
    # To ensure the test remains valid for numerical divergence checking (non-zero output),
    # we add the zeros to the original tensor. This tests if zeros_like correctly
    # propagates dtype/layout in the compiled graph.
    z = torch.zeros_like(t0)
    t3 = t0 + z 

    # Continue with the original logic chain that involves mixed precision
    t4 = t3.norm() # size=(), dtype=bfloat16
    t5 = arg3 # size=(3, 4, 5, 2), dtype=float32
    t6 = t5.var(dim=0) # size=(4, 5, 2), dtype=float32
    t7 = t6.var() # size=(), dtype=float32
    t8 = arg4 # size=(), dtype=float32
    t9 = torch.nn.functional.relu(t8) # size=(), dtype=float32
    t10 = t7 + t4 + t9 # size=(), dtype=float32 (mixed precision addition)
    t11 = torch.pow(torch.pow(t4, t7), t10) # size=(), dtype=float32
    output = t11
    return output

# Setup inputs
# Note: We removed arg1 and arg2 as they are no longer needed with zeros_like
arg0 = torch.rand([5, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg3 = torch.rand([3, 4, 5, 2], dtype=torch.float32, device='cuda', requires_grad=True)
arg4 = torch.rand([], dtype=torch.float32, device='cuda', requires_grad=True)

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0, arg3, arg4)
    out_eager.sum().backward()
    print('Eager Success! ')

    # Compiled execution
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg3, arg4)
    out_compiled.sum().backward()
    print('Compile Success! ')

    # Compare outputs (forward)
    out_eager_sum = out_eager.sum()
    out_compiled_sum = out_compiled.sum()
    diff = (out_eager_sum - out_compiled_sum).abs().item()
    
    # Avoid division by zero
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