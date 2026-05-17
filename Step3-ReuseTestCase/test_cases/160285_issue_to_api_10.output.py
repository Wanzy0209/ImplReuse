import torch
import torch.distributed as dist
import torch.distributed.tensor as dtensor
from torch.distributed.tensor.placement_types import Shard, Replicate

def test_elu_gradients_with_ep():
    """
    Test case to verify gradient correctness in a distributed setting 
    using torch.nn.functional.elu, inspired by Issue 160285.
    
    This test compares gradients between a replicated baseline and an 
    Expert Parallel (EP) sharded scenario to detect gradient doubling bugs.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
    
    # Model dimensions
    in_features = 8
    out_features = 8
    batch_size = 4
    
    # Initialize device mesh
    mesh = dtensor.init_device_mesh(device, (world_size,))
    
    # --- Scenario 1: Replicated Weights (Baseline) ---
    # Weights are replicated across all ranks (simulating Data Parallel or no sharding)
    w_local_rep = torch.randn(out_features, in_features, device=device, requires_grad=True)
    w_dt_rep = dtensor.DTensor(w_local_rep, device_mesh=mesh, placements=[Replicate()])
    
    # Input tensor
    x = torch.randn(batch_size, in_features, device=device)
    
    # Forward pass: Linear -> ELU -> Loss
    # Leveraging torch.nn.functional.elu as the similar API
    y_rep = torch.nn.functional.linear(x, w_dt_rep)
    y_rep = torch.nn.functional.elu(y_rep)
    loss_rep = y_rep.sum()
    
    # Backward pass
    loss_rep.backward()
    
    # --- Scenario 2: Expert Parallel Sharding (Target) ---
    # Weights are sharded on the first dimension (output dimension) across the mesh
    shard_size = out_features // world_size
    w_local_shard = torch.randn(shard_size, in_features, device=device, requires_grad=True)
    w_dt_shard = dtensor.DTensor(w_local_shard, device_mesh=mesh, placements=[Shard(0)])
    
    # Forward pass: Linear -> ELU -> Loss
    y_shard = torch.nn.functional.linear(x, w_dt_shard)
    y_shard = torch.nn.functional.elu(y_shard)
    loss_shard = y_shard.sum()
    
    # Backward pass
    loss_shard.backward()
    
    # --- Verification ---
    # Get local gradients
    grad_rep_local = w_dt_rep.grad.to_local()
    grad_shard_local = w_dt_shard.grad.to_local()
    
    # Extract the slice of the replicated gradient that corresponds to this rank's shard
    start_idx = rank * shard_size
    end_idx = (rank + 1) * shard_size
    grad_rep_slice = grad_rep_local[start_idx:end_idx, :]
    
    # Calculate norms
    norm_rep_slice = torch.norm(grad_rep_slice).item()
    norm_shard = torch.norm(grad_shard_local).item()
    
    print(f"Rank {rank}: Baseline Gradient Norm (slice) = {norm_rep_slice:.6f}")
    print(f"Rank {rank}: Sharded Gradient Norm (local)  = {norm_shard:.6f}")
    
    # The bug report (Issue 160285) indicates that gradients are doubled in EP mode.
    # We assert that the sharded gradient norm is approximately equal to the 
    # corresponding slice of the replicated gradient norm.
    # If the bug is present, norm_shard will be approx 2 * norm_rep_slice.
    
    is_close = torch.isclose(
        torch.tensor(norm_shard), 
        torch.tensor(norm_rep_slice), 
        rtol=0.05, # Allow 5% tolerance for numerical stability
        atol=1e-5
    )
    
    if not is_close:
        ratio = norm_shard / (norm_rep_slice + 1e-9)
        raise AssertionError(
            f"Gradient mismatch on rank {rank}. "
            f"Expected ~{norm_rep_slice:.6f}, got {norm_shard:.6f}. "
            f"Ratio: {ratio:.2f} (Bug likely present: gradients doubled)"
        )
    
    print(f"Rank {rank}: Test passed. Gradients are consistent.")

if __name__ == "__main__":
    # Initialize process group
    # Run with: torchrun --nproc_per_node=2 <script_name>.py
    dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
    test_elu_gradients_with_ep()
    dist.destroy_process_group()