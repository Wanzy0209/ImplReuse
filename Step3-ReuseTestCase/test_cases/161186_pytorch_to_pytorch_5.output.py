import torch
import torch.nn.functional as F

class MyOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor):
        # Create tensors similar in size to the original bug report
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


# Reshape input to be compatible with upsample (requires 3D, 4D, or 5D input)
# 2**20 = 1024 * 1024, so we use shape (1, 1, 1024, 1024)
dummy_input = torch.nn.Parameter(torch.randn(1, 1, 1024, 1024, device="cuda"))

for i in range(1000):
    # Apply the custom autograd function
    flat_out = op_fn(dummy_input.view(-1))
    
    # Reshape output for upsample
    shaped_out = flat_out.view(1, 1, 1024, 1024)
    
    # Replace torch.utils.checkpoint.checkpoint with torch.nn.functional.upsample
    # Verifying if upsample causes memory leaks in a similar loop structure
    full_out = F.upsample(shaped_out, scale_factor=2, mode='nearest')
    
    full_out.sum().backward()
    dummy_input.grad = None  # free gradient memory
    print(i, torch.cuda.memory_allocated() / 1024**2, "MiB")