import torch

# Preserve the configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True

def foo(arg0, arg1):
    # arg0: Input tensor (data)
    # arg1: 0-d tensor (scalar value for quantile)
    # Replicating the pattern of using .item() to pass a scalar to an op
    # Original: t2.fill_diagonal_(t1.item())
    # Adapted: torch.quantile(arg0, arg1.item())
    return torch.quantile(arg0, arg1.item())

# Setup inputs matching the original bug's device and dtype characteristics
# arg0: 2D tensor for quantile operation
arg0 = torch.randn([2, 2], dtype=torch.float32, device='cuda')
# arg1: 0-d tensor representing the quantile (e.g., 0.5)
arg1 = torch.tensor(0.5, dtype=torch.float32, device='cuda')

if __name__ == '__main__':
    # Eager execution
    out_eager = foo(arg0, arg1)
    print('Eager Success! ')

    # Compiled execution
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1)
    print('Compile Success! ')

    # Verify results match
    assert torch.allclose(out_eager, out_compiled), "Divergence between eager and compiled outputs"