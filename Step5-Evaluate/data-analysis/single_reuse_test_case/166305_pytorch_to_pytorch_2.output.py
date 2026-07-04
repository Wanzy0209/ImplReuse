import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo as dynamo
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Configure Dynamo to optimize DDP, which is the context of the bug
dynamo.config.optimize_ddp = True

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))

# Initialize the process group
if dist.is_available():
    dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

class ProdLayer(nn.Module):
    """
    Replaces the custom autograd.Function with the similar API torch.prod.
    This tests if torch.prod works correctly under torch.compile + DDP.
    """
    def forward(self, x):
        # Using torch.prod on the channel dimension
        return torch.prod(x, dim=1, keepdim=True)

def main():
    if LOCAL_RANK == -1:
        print("Skipping test as LOCAL_RANK is not set. Please run with torchrun.")
        return

    device = torch.device(f"cuda:{LOCAL_RANK}")
    
    # Construct model using the ProdLayer containing torch.prod
    model = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1), ProdLayer()).to(device)
    
    # Apply torch.compile
    model = torch.compile(model)
    
    # Wrap with DDP
    model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
    
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    for it in range(3):
        x = torch.rand(2, 3, 256, 256, device=device)
        out = model(x)
        
        # Adjust target shape to match the output of ProdLayer (2, 1, 256, 256)
        target = torch.rand(2, 1, 256, 256, device=device)
        loss = F.mse_loss(out, target)
        
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        
        print(f"[rank{LOCAL_RANK}] iter={it+1} loss={loss.item():.6f}")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()