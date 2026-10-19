# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
import torch.utils.checkpoint

class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor):
        out_0 = torch.zeros(2**20, device=inp.device, dtype=torch.float32)
        out_1 = torch.zeros(2**20, device=inp.device, dtype=torch.float32)
        ctx.save_for_backward(
            inp,
            out_0,
            out_1,
        )
        return out_0, out_1

    @staticmethod
    def backward(ctx, dA, dB):
        _ = ctx.saved_tensors  # this is necessary
        return None


def op_fn(inp):
    return MyOp.apply(inp)[0]


dummy_input = torch.nn.Parameter(torch.randn(2**20, device="cuda"))
for i in range(1000):
    full_out = torch.utils.checkpoint.checkpoint(op_fn, dummy_input, use_reentrant=False)
    full_out.sum().backward()
    dummy_input.grad = None  # free gradient memory
    print(i, torch.cuda.memory_allocated() / 1024**2, "MiB")