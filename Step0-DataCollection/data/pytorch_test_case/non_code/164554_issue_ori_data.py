import itertools
import os
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed._tensor import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy


class TorchGroupedExperts(nn.Module):
    def __init__(self, n_experts: int, hidden_dim: int, ff_dim: int):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(n_experts, hidden_dim, ff_dim))

    def forward(self, x, tokens_per_expert):
        raise NotImplementedError()


# Init distributed
master_addr = os.environ.get("MASTER_ADDR", "localhost")
master_port = os.environ.get("MASTER_PORT", "29500")
os.environ["MASTER_ADDR"] = master_addr
os.environ["MASTER_PORT"] = master_port

world_size = int(os.environ["WORLD_SIZE"])
local_rank = int(os.environ["LOCAL_RANK"])
global_rank = int(os.environ["RANK"])

torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")

# Create device mesh
mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp",))

# Create model on meta device
with torch.device("meta"):
    model = TorchGroupedExperts(n_experts=1, hidden_dim=512, ff_dim=128)

# Apply FSDP2
mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
fully_shard(model, mesh=mesh, mp_policy=mp_policy)

# Verify still on meta
for name, param in model.named_parameters():
    assert param.device == torch.device("meta"), f"{name}: {param.device}"

# Move to device
model.to_empty(device=local_rank)

for name, param in model.named_parameters():
    print(f"Rank {global_rank}: {name} -> {param.device}")

# Forward pass - this is where the bug occurs
x = torch.randn(256, 512, device=f"cuda:{local_rank}", dtype=torch.bfloat16)
tokens_per_expert = torch.tensor([256], device=f"cuda:{local_rank}")
output = model(x, tokens_per_expert)

dist.destroy_process_group()