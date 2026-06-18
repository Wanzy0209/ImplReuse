import os
import torch
import torch.nn as nn
import torch._dynamo as dynamo
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Necessary config for the bug
dynamo.config.optimize_ddp = True

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))
dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

class AllLayer(nn.Module):
    def forward(self, x):
        # Adapted to use torch.all (the similar API)
        # We verify if all elements in the tensor are greater than 0.
        # The result is cast to float to be used in arithmetic operations for the loss calculation.
        is_all_positive = torch.all(x > 0, dim=1, keepdim=True).float()
        return x * is_all_positive

def main():
    device = torch.device(f"cuda:{LOCAL_RANK}")
    # Replaced DoubleLayer with AllLayer to test torch.all
    model = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1), AllLayer()).to(device)
    model = torch.compile(model)
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    for it in range(3):
        # Generate positive random numbers so torch.all returns True
        x = torch.rand(2, 3, 256, 256, device=device)
        out = model(x)
        loss = out.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(f"[rank{LOCAL_RANK}] iter={it+1} loss={loss.item():.6f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()