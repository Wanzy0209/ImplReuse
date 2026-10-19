# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, noop_mask


def run_with_head_count(compiled_fa, H, device, dtype):
    """Run flex attention with a specific head count, creating a captured buffer sized by H."""
    B, S, D = 2, 256, 64

    # Create captured buffer that depends on dynamic H
    head_scale = torch.randn(H, device=device, dtype=dtype, requires_grad=True)

    def score_mod(score, batch, head, token_q, token_kv):
        return score * head_scale[head]

    print(f"  Running with H={H}, head_scale.shape={head_scale.shape}")

    # Run multiple iterations with the same head_scale
    for i in range(5):
        q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)

        block_mask = create_block_mask(noop_mask, B, 1, S, S, device=device)

        outputs = compiled_fa(q, k, v, score_mod=score_mod, block_mask=block_mask)
        loss = outputs.sum()
        loss.backward()

    print(f"  ✓ Completed {i+1} iterations")


def main():
    device = "cuda"
    dtype = torch.float16
    torch.manual_seed(0)

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