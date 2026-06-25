import torch
import torch.distributed as dist

def reproduce_moe_error():
    # Simplified MoE routing scenario
    rank = dist.get_rank() if dist.is_initialized() else 0
    tokens = torch.randn(16384, 512) if rank == 0 else torch.zeros(0, 512)
    
    # Simulate graph reuse with dynamic token counts
    try:
        compiled_fn = torch.compile(lambda x: x.sum(dim=0))
        result = compiled_fn(tokens)
        assert result.size(0) == 16384  # Fails when tokens is empty
    except RuntimeError as e:
        print(f"Rank {rank}: {e}")

if __name__ == "__main__":
    reproduce_moe_error()