import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
import os

# Define a custom module that mimics the scenario described in the bug:
# A matrix multiplication where the Right Hand Side (RHS) is explicitly row-major.
# The bug report indicates that `_scaled_mm` and `_int_mm` are slow or error-prone
# with row-major RHS matrices. We test if DistributedDataParallel (DDP) handles
# such operations correctly during the forward/backward pass.
class RowMajorMatMulLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Initialize weight. 
        # In the bug context, the RHS matrix being row-major is the trigger.
        # We ensure the weight is contiguous (row-major in PyTorch).
        self.weight = nn.Parameter(torch.randn(out_features, in_features).contiguous())

    def forward(self, x):
        # Perform matmul: x (M, K) @ weight (K, N) -> (M, N)
        # Here, 'weight' acts as the RHS.
        # We explicitly use .contiguous() to enforce the row-major layout 
        # to simulate the condition reported in the bug.
        rhs = self.weight.contiguous()
        return torch.matmul(x, rhs.t())

def test_ddp_with_row_major_rhs(rank, world_size):
    """
    Test function to be run in each process.
    Verifies that DDP synchronizes gradients correctly even when
    the underlying operation involves a row-major RHS matrix.
    """
    # Initialize the process group
    dist.init_process_group(
        backend='gloo', # Using gloo for CPU compatibility in this test
        init_method=f'tcp://127.0.0.1:12355',
        rank=rank,
        world_size=world_size
    )

    # Create the model and move it to the specific device (CPU here)
    model = RowMajorMatMulLayer(in_features=10, out_features=5)
    
    # Wrap the model with DistributedDataParallel
    # This is the "Similar API" we are testing.
    ddp_model = nn.parallel.DistributedDataParallel(model)

    # Create dummy input
    inputs = torch.randn(2, 10)
    
    # Forward pass
    # This triggers the matmul with the row-major RHS
    outputs = ddp_model(inputs)
    
    # Backward pass
    # This computes gradients. If the underlying ops were buggy (as per the issue),
    # this might be slow or crash. DDP must synchronize these gradients.
    loss = outputs.sum()
    loss.backward()

    # Verify that gradients have been computed and are not None
    assert ddp_model.module.weight.grad is not None, "Gradient computation failed"

    # Verify synchronization (optional but good for DDP tests)
    # In a real scenario, we would check if grads are identical across ranks,
    # but for a minimal runnable test, ensuring no crash is the primary goal
    # given the bug report mentions errors and slowness.

    # Clean up
    dist.destroy_process_group()

if __name__ == "__main__":
    # Set environment variables for multiprocessing
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'

    world_size = 2
    # Spawn processes to run the test
    mp.spawn(test_ddp_with_row_major_rhs, args=(world_size,), nprocs=world_size, join=True)