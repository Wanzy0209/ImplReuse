import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Setup for distributed environment
os.environ["TORCH_LOGS"] = "output_code"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "29500"

def f(x, y):
    y2 = torch.cat(
        [
            x[:, 1:],
            y[:, None] + 32 * 2048,
        ],
        dim=1,
    )

    x2 = x[:, 1:, None]
    y3 = y2[:, -1:, None]

    return (
        torch.cat([x2, y3], dim=1)
        + torch.arange(-2048, 0, device=x.device)[None, None, :]
    ).reshape(1, 32 * 2048)

def worker(rank, size):
    # Initialize process group
    dist.init_process_group("gloo", rank=rank, world_size=size)
    
    # Use CPU for simplicity in this test case to avoid NCCL requirements
    device = "cpu"

    # Original inputs
    x = torch.zeros(1, 32, dtype=torch.int64, device=device)
    y = torch.zeros(1, dtype=torch.int32, device=device)

    result = f(x, y)
    
    # Check if torch.compile is available (introduced in PyTorch 2.0)
    if hasattr(torch, "compile"):
        # Adapted call site: using torch.distributed.reduce
        # We wrap the logic in torch.compile to check if the bug affects this API
        # when compiled, similar to the original report.
        compiled_func = torch.compile(lambda t: dist.reduce(t, dst=0))
        compiled_func(result)
    else:
        # Fallback for older PyTorch versions: run the operation without compilation
        # to maintain the core test logic of performing the reduce operation.
        dist.reduce(result, dst=0)

    dist.destroy_process_group()

if __name__ == "__main__":
    size = 2
    mp.spawn(worker, args=(size,), nprocs=size, join=True)