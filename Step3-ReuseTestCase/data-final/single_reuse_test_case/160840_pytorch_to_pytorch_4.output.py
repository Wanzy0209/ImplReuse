import torch
import numpy as np

torch.manual_seed(0)
torch._inductor.config.fallback_random = True

def foo(input, target):
    # Adapted to use torch.nn.functional.kl_div instead of interpolate
    # Using reduction='none' to maintain tensor shape for subsequent operations
    kl = torch.nn.functional.kl_div(
        input,
        target,
        reduction='none'
    )
    squeeze = kl.squeeze(0)
    argmin = squeeze.argmin(1)
    return argmin

np.random.seed(0)
# Generate inputs for kl_div (input and target)
x = np.random.uniform(0, 10, size=(1, 40, 1, 1))  # dtype = float64
y = np.random.uniform(0, 10, size=(1, 40, 1, 1))  # dtype = float64

cfoo = torch.compile(foo)
eager_res = foo(torch.from_numpy(x), torch.from_numpy(y))
compile_res = cfoo(torch.from_numpy(x), torch.from_numpy(y))
torch.testing.assert_close(eager_res, compile_res)