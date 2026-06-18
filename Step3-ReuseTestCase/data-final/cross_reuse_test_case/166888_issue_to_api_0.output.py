import torch

def f(x, scale):
    # Reproduce the bug pattern: calling .item() on a float tensor argument
    # inside a compiled function, adapted to use the similar API torch.tanh
    y = torch.tanh(x * scale.item())
    return y

# Compile with the same settings as the original bug report
compiled_func = torch.compile(f, backend='inductor', fullgraph=True)

# Setup inputs on CUDA
x = torch.randn(10, 20, 30, device='cuda')
scale = torch.tensor(2.0, device='cuda')

# Execute the compiled function
# If the bug exists, this will raise torch._inductor.exc.InductorError: NameError: 'zuf0' is not defined
compiled_func(x, scale)