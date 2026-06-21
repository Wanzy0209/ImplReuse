"""
Run:

torchrun --nproc_per_node=2 repro.py
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist

class ProdModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 32)

    def forward(self, x):
        x = self.linear(x)
        # Test torch.prod as the similar API
        return torch.prod(x, dim=1)

def main() -> None:
    # Setup device using standard distributed API (compatible with older PyTorch versions)
    # This replaces the missing torch.distributed.device_mesh module
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    model = ProdModel().to(device)
    model.train()

    # Create dummy input
    input_tensor = torch.randn(8, 32, device=device)

    # Forward pass
    output = model(input_tensor)

    # Verify torch.prod behavior
    # Expected output shape: (8,) because we reduced dim=1
    assert output.shape == (8,), f"Expected shape (8,), got {output.shape}"
    
    # Verify the calculation is correct
    with torch.no_grad():
        expected_output = torch.prod(model.linear(input_tensor), dim=1)
        assert torch.allclose(output, expected_output), "torch.prod calculation mismatch"

    print(f"Rank {local_rank}: torch.prod test passed.")

if __name__ == "__main__":
    main()