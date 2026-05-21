# ddp_compile_broadcast_test.py
import os, torch
import torch.nn as nn, torch.nn.functional as F
import torch._dynamo as dynamo
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

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