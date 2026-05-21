import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo as dynamo
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Bug configuration
dynamo.config.optimize_ddp = True

# Distributed setup
LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

# Define the custom operator namespace and name
NAMESPACE = "test_ddp_compile"
OP_NAME = "simple_double"
QUALNAME = f"{NAMESPACE}::{OP_NAME}"

# 1. Define the custom operator schema
# We use a try-except block to allow re-running in interactive environments if necessary
try:
    torch.library.define(QUALNAME, "(Tensor x) -> Tensor")
except AssertionError:
    pass  # Op might already be defined

# 2. Register the Fake Implementation (Abstract Impl)
# This uses the similar API: torch.library.register_fake (aliased as torch.library.impl_abstract)
# This explicitly tells torch.compile the output metadata without running the function.
@torch.library.register_fake(QUALNAME)
def simple_double_fake(x):
    return torch.empty_like(x)

# 3. Register the actual implementation
@torch.library.impl(QUALNAME, "CompositeExplicitAutograd")
def simple_double_impl(x):
    return x * 2

# 4. Define the custom autograd function using the library op
# This preserves the structure of the original bug (custom autograd logic)
# while utilizing the registered abstract implementation.
class SimplistDoubleFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.ops.test_ddp_compile.simple_double(x)

    @staticmethod
    def backward(ctx, grad_out):
        # Custom backward logic
        return grad_out * 2

class DoubleLayer(nn.Module):
    def forward(self, x):
        return SimplistDoubleFn.apply(x)

def main():
    # Initialize process group
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() and dist.is_nccl_available() else "gloo"
        if backend == "nccl":
            dist.init_process_group(backend=backend)
        else:
            # Fallback for non-cuda environments to allow script execution
            # Note: DDP behavior is specific to distributed context
            pass

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{LOCAL_RANK}")
    else:
        device = torch.device("cpu")

    model = nn.Sequential(nn.Conv2d(3, 3, 3, padding=1), DoubleLayer()).to(device)
    
    # Compile the model
    model = torch.compile(model)
    
    # Wrap with DDP
    if torch.cuda.is_available() and dist.is_initialized():
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=True)
    
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    for it in range(3):
        x = torch.rand(2, 3, 256, 256, device=device)
        out = model(x)
        loss = F.mse_loss(out, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        print(f"[rank{LOCAL_RANK}] iter={it+1} loss={loss.item():.6f}")

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()