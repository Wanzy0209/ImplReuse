# ddp_compile_broadcast_test.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Fix: Handle missing torch._dynamo gracefully
try:
    import torch._dynamo as dynamo
    HAS_DYNAMO = True
except (ImportError, ModuleNotFoundError):
    dynamo = None
    HAS_DYNAMO = False
    print("Warning: torch._dynamo is not available. Skipping dynamo-specific configurations and compilation.")

if HAS_DYNAMO:
    dynamo.config.optimize_ddp = True  # <== necessary for the bug context

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

class BroadcastLayer(nn.Module):
    def forward(self, x):
        # Adaptation: Use torch.distributed.broadcast_object_list
        # We broadcast a list containing a scalar derived from the tensor to verify the API
        # Rank 0 sends the mean, others send dummy data
        obj_list = [x.mean().item()] if LOCAL_RANK == 0 else [0.0]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        # Return x to maintain tensor flow for loss calculation
        return x

def main():
    device = torch.device(f"cuda:{LOCAL_RANK}")
    # Use BroadcastLayer instead of DoubleLayer
    model = nn.Sequential(nn.Conv2d(3,3,3,padding=1), BroadcastLayer()).to(device)

    # Fix: Only compile if torch._dynamo is available
    if HAS_DYNAMO and hasattr(torch, "compile"):
        model = torch.compile(model)

    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    for it in range(3):
        x = torch.rand(2,3,256,256, device=device)
        out = model(x)
        loss = F.mse_loss(out, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(f"[rank{LOCAL_RANK}] iter={it+1} loss={loss.item():.6f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()