import os
import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.experimental import implicit_replication
from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker


class RMSNormLinearModule(nn.Module):
    """
    Module combining RMSNorm and Linear to reproduce the KeyError scenario.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.output(x)
        return x


def test_fsdp_mem_tracker_rmsnorm():
    """
    Test case for Issue 164663: FSDPMemTracker fails with KeyError for RMSNorm during backward.
    
    This test preserves the original bug reproduction logic (sharding RMSNorm with Linear)
    while adopting a device handling pattern similar to tf.data.experimental.copy_to_device
    (explicit source and target device variables) to ensure clarity in device placement.
    """
    # Explicit device handling inspired by the similar API pattern
    target_device = "cuda:0"
    source_device = "cpu"

    # Setup distributed environment
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

    d_model = 128

    # Initialize model on source device
    model = RMSNormLinearModule(d_model)
    
    # Move to target device
    model = model.to(target_device)
    
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    # Apply sharding: Wrapping RMSNorm and Linear in a single fully_shard call
    # is the specific trigger for the KeyError in the bug report.
    fully_shard([model.norm, model.output], mesh=mesh)
    fully_shard(model, mesh=mesh)

    tracker = FSDPMemTracker(model)

    # Execute forward and backward pass
    # The bug manifests during the backward pass (pre-backward hook)
    with tracker, implicit_replication():
        x = torch.randn(16, d_model, device=target_device)
        y = model(x)
        loss = y.sum()
        
        # This line is expected to raise KeyError in the buggy version
        loss.backward()

    print("Test Passed: FSDPMemTracker successfully tracked RMSNorm during backward.")


if __name__ == "__main__":
    test_fsdp_mem_tracker_rmsnorm()