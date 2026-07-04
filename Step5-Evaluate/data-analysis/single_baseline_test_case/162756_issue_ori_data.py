# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
torch._inductor.config.combo_kernels = True

@torch.compile
def fn(x, y, z):
    return x.sum(1), y.mean(1), z.cumsum(1)

inps = (
    torch.rand(16, 128, device="cuda"),
    torch.rand(32, 128, device="cuda"),
    torch.rand(32, 256, device="cuda"),
)

fn(*inps)