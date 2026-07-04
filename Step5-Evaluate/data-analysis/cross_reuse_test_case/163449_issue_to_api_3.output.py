import torch
import sys

# Preserve the specific configurations that triggered the original bug
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    # arg0, arg1: bfloat16 inputs for logaddexp (replacing addmm inputs)
    # arg2: float32 input for variance calculation (to mix types like the original)

    # Replace torch.addmm with torch.logaddexp
    # Original: t3 = torch.addmm(t0, t1, t2) # size=(5, 4)
    # New: t3 = torch.logaddexp(arg0, arg1)  # size=(5, 4)
    t3 = torch.logaddexp(arg0, arg1)

    # Keep the reduction logic to test scalar output precision
    # Original: t4 = t3.norm()
    t4 = t3.norm()

    # Keep the variance logic to mix float32 operations
    # Original: t5 = arg3; t6 = t5.var(dim=0); t7 = t6.var()
    t5 = arg2
    t6 = t5.var(dim=0)
    t7 = t6.var()

    # Combine results (omitting the scalar relu from original for minimalism)
    # Original: t10 = t7 + t4 + t9
    t10 = t7 + t4

    # Keep the power operation which is sensitive to precision
    # Original: t11 = torch.pow(torch.pow(t4, t7), t10)
    t11 = torch.pow(torch.pow(t4, t7), t10)

    output = t11
    return output

# Generate inputs
# Using bfloat16 for the main operation inputs, similar to the original bug
arg0 = torch.rand([5, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg1 = torch.rand([5, 4], dtype=torch.bfloat16, device='cuda', requires_grad=True)
# Using float32 for the secondary operation
arg2 = torch.rand([3, 4, 5, 2], dtype=torch.float32, device='cuda', requires_grad=True)

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
    rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100
    print(f'Relative diff (sum): {rel_diff:.6f}%')

    if rel_diff > 5:
        print(f' Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', out_eager_sum.item())
        print('out_compiled_sum:', out_compiled_sum.item())
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)