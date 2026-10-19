import torch
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, noop_mask


def run_with_head_count(compiled_fa, H, device, dtype):
    """Run flex attention with a specific head count, creating a captured buffer sized by H."""
    B, S, D = 2, 256, 64

    # Replacement: torch.randn -> torch.nn.functional.max_pool1d
    # Create captured buffer that depends on dynamic H using max_pool1d
    # Input shape (1, 1, H+2) -> Output shape (1, 1, H) -> Squeeze to (H,)
    dummy_h = torch.ones(1, 1, H + 2, device=device, dtype=dtype)
    pooled_h = F.max_pool1d(dummy_h, kernel_size=3, stride=1)
    head_scale = pooled_h.squeeze().requires_grad_(True)

    def score_mod(score, batch, head, token_q, token_kv):
        return score * head_scale[head]

    print(f"  Running with H={H}, head_scale.shape={head_scale.shape}")

    # Run multiple iterations with the same head_scale
    for i in range(5):
        # Replacement: torch.randn -> torch.nn.functional.max_pool1d
        # Create q using max_pool1d. Target shape (B, H, S, D).
        # Reshape to (B*H, S, D+2) for pooling, then reshape back to (B, H, S, D).
        dummy_q = torch.ones(B * H, S, D + 2, device=device, dtype=dtype)
        pooled_q = F.max_pool1d(dummy_q, kernel_size=3, stride=1)
        q = pooled_q.reshape(B, H, S, D).requires_grad_(True)

        # Replacement: torch.randn_like -> torch.dist
        # Create k using torch.dist. 
        # k = q + dist(q, zeros) to maintain shape and use the API.
        k = q + torch.dist(q, torch.zeros_like(q))
        k.requires_grad_(True)

        # Replacement: torch.randn_like -> torch.dist
        # Create v using torch.dist.
        # v = q + dist(q, ones) to maintain shape and use the API.
        v = q + torch.dist(q, torch.ones_like(q))
        v.requires_grad_(True)

        block_mask = create_block_mask(noop_mask, B, 1, S, S, device=device)

        outputs = compiled_fa(q, k, v, score_mod=score_mod, block_mask=block_mask)
        loss = outputs.sum()
        loss.backward()

    print(f"   Completed {i+1} iterations")


def main():
    device = "cuda"
    dtype = torch.float16

    # Replacement: torch.manual_seed -> torch.isnan
    # Original: torch.manual_seed(0)
    # New: Check isnan on a dummy tensor to satisfy API replacement
    _ = torch.isnan(torch.tensor([0.0]))

    # Test with different head counts - this makes H a dynamic dimension
    # and the captured buffer (head_scale) changes size with H
    head_counts = [4, 8, 4, 16, 4]

    compiled_fa = torch.compile(flex_attention, fullgraph=True, dynamic=True)

    print(f"Running flex-attention with dynamic head counts on {device}, dtype={dtype}")
    print(f"Testing head counts: {head_counts}\n")

    for iteration, H in enumerate(head_counts, start=1):
        print(f"Iteration {iteration}:")
        run_with_head_count(compiled_fa, H, device, dtype)


if __name__ == "__main__":
    main()