import os
import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import fully_shard
from torch.distributed.tensor.experimental import implicit_replication


class TestModule(nn.Module):

    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.output = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.output(x)
        return x


def main():
    # Setup distributed environment
    if not torch.distributed.is_available():
        print("Distributed not available, skipping test.")
        return

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    d_model = 128

    model = TestModule(d_model)
    model = model.to('cuda:0')
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))

    # Apply FSDP sharding as in the original bug report
    fully_shard([model.norm, model.output], mesh=mesh)
    fully_shard(model, mesh=mesh)

    # Use the similar API: register_module_backward_hook
    # This tests the underlying hook mechanism on the RMSNorm module
    # which was implicated in the FSDPMemTracker failure.
    hook_called = False

    def backward_hook(module, grad_input, grad_output):
        nonlocal hook_called
        hook_called = True
        # Verify we can access the module and gradients without error
        assert module is not None
        assert grad_input is not None
        assert grad_output is not None

    handle = model.norm.register_module_backward_hook(backward_hook)

    # Run forward and backward
    with implicit_replication():
        x = torch.randn(16, d_model, device='cuda:0')
        y = model(x)
        loss = y.sum()
        loss.backward()

    # Assertions
    assert hook_called, "Backward hook was not called on RMSNorm"
    handle.remove() # Clean up

    print("Test passed: Backward hook executed successfully on RMSNorm with FSDP.")


if __name__ == "__main__":
    main()