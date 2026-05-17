import torch
import numpy as np

torch.manual_seed(0)
torch._inductor.config.fallback_random = True

def foo(input):
    # Adapted to use torch.nn.functional.upsample_nearest
    # Note: align_corners, recompute_scale_factor, and antialias are removed 
    # as they are not arguments for upsample_nearest.
    upsample = torch.nn.functional.upsample_nearest(
        input,
        size=[40, 40],
        scale_factor=None
    )
    squeeze = upsample.squeeze(0)
    argmin = squeeze.argmin(1)
    return argmin


np.random.seed(0)
x = np.random.uniform(0, 10, size=(1, 40, 1, 1))  # dtype = float64

cfoo = torch.compile(foo)
eager_res = foo(torch.from_numpy(x))
compile_res = cfoo(torch.from_numpy(x))

torch.testing.assert_close(eager_res, compile_res)