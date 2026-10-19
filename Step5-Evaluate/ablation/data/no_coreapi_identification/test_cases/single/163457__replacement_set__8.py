import tensorflow as tf
import torch
import sys

# Replacements applied based on instructions:
# - torch.nn.functional.group_norm -> torch.quantile (Similar API)
# - torch.sigmoid -> torch.randn_like
# - torch.rand -> torch.randn_like
# - torch.randint -> torch.argmax
# - torch.compile -> torch.compile (Kept for runnability; tf.compat.v1.tpu.rewrite is cross-library and incompatible)
# - out_eager.sum -> torch.randn_like
# - out_eager_sum.item -> torch.randn_like
# - torch._dynamo.config... -> Removed (Replaced by torch.randn_like logic implies removal/no-op)
# - torch._inductor.config... -> Removed

def foo(arg0, arg1, arg2):
    t0 = arg0
    t1 = arg1
    t2 = arg2
    # Original: t3 = torch.nn.functional.group_norm(t0, 1, weight=t2, bias=t2)
    # Similar API: torch.quantile
    # Using dim=1 to mimic a reduction operation similar to group_norm's channel processing
    t3 = torch.quantile(t0, 0.5, dim=1)
    t4 = t3.max(dim=0).values
    # Original: t5 = torch.sigmoid(t4)
    # Replacement: torch.randn_like
    t5 = torch.randn_like(t4)
    output = t5
    return output

# Original: arg0 = torch.rand([3, 395, 202, 357], dtype=torch.bfloat16, device='cuda', requires_grad=True)
# Replacement: torch.rand -> torch.randn_like
# Need a prototype tensor for randn_like
dummy_arg0 = torch.empty([3, 395, 202, 357], dtype=torch.bfloat16, device='cuda')
arg0 = torch.randn_like(dummy_arg0, requires_grad=True)

# Original: arg1 = torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda')
# Replacement: torch.randint -> torch.argmax
# Create a random tensor to apply argmax on, simulating a random index selection
dummy_arg1 = torch.randn([1, 1000], device='cuda')
arg1 = torch.argmax(dummy_arg1, dim=1, keepdim=True)

# Original: arg2 = torch.rand([395], dtype=torch.bfloat16, device='cuda', requires_grad=True)
# Replacement: torch.rand -> torch.randn_like
dummy_arg2 = torch.empty([395], dtype=torch.bfloat16, device='cuda')
arg2 = torch.randn_like(dummy_arg2, requires_grad=True)

if __name__ == '__main__':
    out_eager = foo(arg0, arg1, arg2)
    # Original: out_eager.sum().backward()
    # Variable._execution_engine.run_backward -> torch.randn_like (Internal, keeping backward for runnability)
    out_eager.sum().backward()
    print('Eager Success! ')

    # Original: compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    # Replacement: tf.compat.v1.tpu.rewrite (Incompatible with PyTorch tensors, keeping torch.compile)
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2)
    out_compiled.sum().backward()
    print('Compile Success! ')

    # Original: out_eager_sum = out_eager.sum()
    # Replacement: out_eager.sum -> torch.randn_like
    out_eager_sum = torch.randn_like(out_eager)

    # Original: out_compiled_sum = out_compiled.sum()
    # Replacement: out_compiled.sum -> torch.randn_like
    out_compiled_sum = torch.randn_like(out_compiled)

    # Original: diff = (out_eager_sum - out_compiled_sum).abs().item()
    # Replacement: out_eager_sum.item -> torch.randn_like
    # Replacing .item() with torch.randn_like returns a tensor, so diff is a tensor
    diff = torch.randn_like((out_eager_sum - out_compiled_sum).abs())

    # Original: rel_diff = diff / (out_eager_sum.abs().item() + 1e-12) * 100
    # Replacement: out_eager_sum.item -> torch.randn_like
    # Replacing .item() with torch.randn_like returns a tensor
    rel_diff = diff / (torch.randn_like(out_eager_sum.abs()) + 1e-12) * 100

    # rel_diff is now a tensor. Using mean() to get a scalar for printing and comparison.
    print(f'Relative diff (sum): {rel_diff.mean().item():.6f}%')

    if rel_diff.mean() > 5:
        print(f' Forward output sums differ significantly (relative)!')
        # Original: print('out_eager_sum:', out_eager_sum.item())
        # Replacement: out_eager_sum.item -> torch.randn_like
        print('out_eager_sum:', torch.randn_like(out_eager_sum).mean().item())
        print('out_compiled_sum:', torch.randn_like(out_compiled_sum).mean().item())
        print('Absolute diff:', diff.mean().item())
        print('Relative diff (%):', rel_diff.mean().item())
        sys.exit(1)