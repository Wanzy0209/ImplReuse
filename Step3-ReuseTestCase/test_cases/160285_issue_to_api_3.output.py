import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
import os

def setup():
    """Initialize the distributed environment."""
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())

def cleanup():
    """Cleanup the distributed environment."""
    dist.destroy_process_group()

def test_transformer_encoder_gradients():
    """
    Test case to verify gradient norms in a TransformerEncoder when using 
    DTensor for weight sharding (simulating parallelism similar to the MoE issue).
    
    This test adapts the logic from the MoE gradient bug report to the 
    torch.nn.TransformerEncoder API.
    """
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    print(f"[Rank {rank}] Starting TransformerEncoder Gradient Test")

    # 1. Define the Model
    # Using a small TransformerEncoder for testing
    d_model = 64
    nhead = 4
    num_layers = 2
    encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
    model = nn.TransformerEncoder(encoder_layer, num_layers=num_layers).to(device)

    # 2. Shard the model weights using DTensor
    # This simulates the "Expert Parallel" sharding logic from the bug report
    # where weights are distributed across the device mesh.
    mesh = init_device_mesh("cuda", (world_size,))
    
    print(f"[Rank {rank}] Applying DTensor sharding to TransformerEncoder weights...")
    
    for name, param in model.named_parameters():
        # Shard along the first dimension (row parallelism) for demonstration
        # In the original bug, specific sharding strategies caused gradient doubling.
        local_param = param.data
        dist_param = DTensor.from_local(local_param, mesh, [Shard(0)])
        # Replace the parameter data with the DTensor
        param.data = dist_param

    # 3. Create Input Data
    # Batch size 8, Sequence length 10, Embedding dim 64
    # We also distribute the input if necessary, but here we keep input local 
    # to see how the sharded weights handle it, or distribute input as well.
    # For consistency with typical parallel training, let's distribute input.
    batch_size = 8
    seq_len = 10
    local_input = torch.randn(batch_size, seq_len, d_model, device=device)
    dist_input = DTensor.from_local(local_input, mesh, [Shard(0)]) # Shard batch dim

    # 4. Forward Pass
    print(f"[Rank {rank}] Running forward pass...")
    output = model(dist_input)
    
    # 5. Calculate Loss
    # Simple sum loss to ensure gradients flow
    loss = output.sum()
    
    # 6. Backward Pass
    print(f"[Rank {rank}] Running backward pass...")
    loss.backward()

    # 7. Analyze Gradients
    # The original bug reported gradients were doubled (2x size).
    # We check the gradient norms here.
    print(f"\n[Rank {rank}] ==================================================")
    print(f"[Rank {rank}] TRANSFORMER ENCODER GRADIENT ANALYSIS (Sharded)")
    print(f"[Rank {rank}] ==================================================")
    
    total_grad_norm = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            # param.grad is a DTensor
            grad_dtensor = param.grad
            local_grad = grad_dtensor.to_local()
            
            grad_norm = local_grad.norm().item()
            total_grad_norm += grad_norm ** 2
            
            print(f"[Rank {rank}]   {name}:")
            print(f"      Global Shape: {param.shape}, Local Shape: {local_grad.shape}")
            print(f"      Grad Norm: {grad_norm:.6f}")
            
            # Check for NaN or Inf (basic sanity check)
            assert not torch.isnan(local_grad).any(), f"NaN detected in gradients for {name}"
            assert not torch.isinf(local_grad).any(), f"Inf detected in gradients for {name}"
            
            # In the original bug, gradients were exactly 2x.
            # Without a baseline (non-sharded run) in the same process, we can't assert the exact value,
            # but we can print it for manual verification similar to the original issue.
            
    total_grad_norm = total_grad_norm ** 0.5
    print(f"[Rank {rank}]   Total gradient norm: {total_grad_norm:.6f}")
    print(f"[Rank {rank}] ==================================================\n")

if __name__ == "__main__":
    # Check if running with torchrun
    if "RANK" not in os.environ and "WORLD_SIZE" not in os.environ:
        print("This test must be run with torchrun, e.g.:")
        print("torchrun --nproc_per_node=2 test_transformer_gradients.py")
        exit(1)

    setup()
    try:
        test_transformer_encoder_gradients()
    finally:
        cleanup()