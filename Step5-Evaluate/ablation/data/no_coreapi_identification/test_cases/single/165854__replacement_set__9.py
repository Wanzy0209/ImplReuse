import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, noop_mask


def run_with_head_count(compiled_fa, H, device, dtype):
    """Run flex attention with a specific head count, creating a captured buffer sized by H."""
    B, S, D = 2, 256, 64

    # Replacement: torch.randn -> torch.nn.functional.hinge_embedding_loss
    # Create captured buffer that depends on dynamic H
    # Using reduction='none' to maintain the tensor dimension H
    head_scale = F.hinge_embedding_loss(
        torch.zeros(H, device=device, dtype=dtype, requires_grad=True),
        torch.ones(H, device=device, dtype=dtype),
        reduction='none'
    )

    def score_mod(score, batch, head, token_q, token_kv):
        return score * head_scale[head]

    print(f"  Running with H={H}, head_scale.shape={head_scale.shape}")

    # Run multiple iterations with the same head_scale
    for i in range(5):
        # Replacement: torch.randn -> torch.nn.functional.hinge_embedding_loss
        # q needs to be (B, H, S, D). We generate a flat tensor and view it.
        q_flat = F.hinge_embedding_loss(
            torch.zeros(B * H * S * D, device=device, dtype=dtype, requires_grad=True),
            torch.ones(B * H * S * D, device=device, dtype=dtype),
            reduction='none'
        )
        q = q_flat.view(B, H, S, D)

        # Replacement: torch.randn_like -> torch.nn.Conv2d
        # k needs to be (B, H, S, D). We use Conv2d to generate k from q.
        # Mapping (B, H, S, D) to (N, C, H, W) for Conv2d: (B, H, S, D)
        conv_k = nn.Conv2d(H, H, kernel_size=1, bias=False).to(device)
        k = conv_k(q)

        # Replacement: torch.randn_like -> torch.nn.Conv2d
        # v needs to be (B, H, S, D). We use Conv2d to generate v from q.
        conv_v = nn.Conv2d(H, H, kernel_size=1, bias=False).to(device)
        v = conv_v(q)

        block_mask = create_block_mask(noop_mask, B, 1, S, S, device=device)

        outputs = compiled_fa(q, k, v, score_mod=score_mod, block_mask=block_mask)
        loss = outputs.sum()
        loss.backward()

    print(f"   Completed {i+1} iterations")


def main():
    device = "cuda"
    dtype = torch.float16
    
    # Replacement: torch.manual_seed -> torch.rand_like
    # manual_seed(0) is replaced with a call to rand_like on a dummy tensor
    _ = torch.rand_like(torch.tensor(0.0))

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