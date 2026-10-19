import torch
import tensorflow as tf
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, noop_mask
import torch.nn.functional as F


def run_with_head_count(compiled_fa, H, device, dtype):
    """Run flex attention with a specific head count, creating a captured buffer sized by H."""
    B, S, D = 2, 256, 64

    # Replacement: torch.randn -> torch.nn.functional.cosine_embedding_loss
    # Create captured buffer that depends on dynamic H using cosine_embedding_loss
    # We construct dummy inputs to generate a tensor of shape (H,)
    inp1 = torch.ones(H, D, device=device, dtype=dtype, requires_grad=True)
    inp2 = torch.zeros(H, D, device=device, dtype=dtype, requires_grad=True)
    target = torch.ones(H, device=device, dtype=torch.long)
    head_scale = F.cosine_embedding_loss(inp1, inp2, target, reduction='none')

    def score_mod(score, batch, head, token_q, token_kv):
        return score * head_scale[head]

    print(f"  Running with H={H}, head_scale.shape={head_scale.shape}")

    # Run multiple iterations with the same head_scale
    for i in range(5):
        # Replacement: torch.randn -> torch.nn.functional.cosine_embedding_loss
        # Generate q using cosine_embedding_loss. 
        # Reshape logic: (B, H, S, D) -> (N, D) -> Loss(N) -> (B, H, S, D)
        N = B * H * S
        q_inp1 = torch.ones(N, D, device=device, dtype=dtype, requires_grad=True)
        q_inp2 = torch.zeros(N, D, device=device, dtype=dtype, requires_grad=True)
        q_target = torch.ones(N, device=device, dtype=torch.long)
        q_flat = F.cosine_embedding_loss(q_inp1, q_inp2, q_target, reduction='none')
        q = q_flat.view(B, H, S, D)

        # Replacement: torch.randn_like -> torch.median
        # Generate k and v using torch.median on q, expanding to match shape
        k_val, _ = torch.median(q, dim=1, keepdim=True)
        k = k_val.expand(B, H, S, D)
        k.requires_grad_(True)

        v_val, _ = torch.median(q, dim=2, keepdim=True)
        v = v_val.expand(B, H, S, D)
        v.requires_grad_(True)

        block_mask = create_block_mask(noop_mask, B, 1, S, S, device=device)

        outputs = compiled_fa(q, k, v, score_mod=score_mod, block_mask=block_mask)
        loss = outputs.sum()
        loss.backward()

    print(f"   Completed {i+1} iterations")


def main():
    device = "cuda"
    dtype = torch.float16

    # Replacement: torch.manual_seed -> tf.compat.v1.assert_variables_initialized
    # Passing an empty list as we don't have TF variables in this context
    tf.compat.v1.assert_variables_initialized([])

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