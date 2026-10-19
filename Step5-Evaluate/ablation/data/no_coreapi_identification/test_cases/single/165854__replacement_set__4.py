import torch
import torch.nn.functional as F
import tensorflow as tf
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, noop_mask

# Create a dummy source tensor for adaptive_max_pool2d replacements.
# This is necessary because adaptive_max_pool2d requires an input tensor,
# whereas torch.randn generates data from scratch.
# We make it large enough to be downsampled to various sizes in the test.
_DUMMY_POOL_SRC = torch.randn(1, 1, 1024, 1024, device="cuda", dtype=torch.float16, requires_grad=True)


def run_with_head_count(compiled_fa, H, device, dtype):
    """Run flex attention with a specific head count, creating a captured buffer sized by H."""
    B, S, D = 2, 256, 64

    # Original: head_scale = torch.randn(H, device=device, dtype=dtype, requires_grad=True)
    # Replacement: torch.nn.functional.adaptive_max_pool2d
    # We pool the dummy source to shape (1, 1, H, 1) and squeeze to get (H,)
    pooled = F.adaptive_max_pool2d(_DUMMY_POOL_SRC, (H, 1))
    head_scale = pooled.squeeze().requires_grad_(True)

    def score_mod(score, batch, head, token_q, token_kv):
        return score * head_scale[head]

    print(f"  Running with H={H}, head_scale.shape={head_scale.shape}")

    # Run multiple iterations with the same head_scale
    for i in range(5):
        # Original: q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        # Replacement: torch.nn.functional.adaptive_max_pool2d
        # Create a dummy input for q generation and pool it to (S, D)
        dummy_q_src = torch.randn(B, H, 512, 512, device=device, dtype=dtype, requires_grad=True)
        q = F.adaptive_max_pool2d(dummy_q_src, (S, D))

        # Original: k = torch.randn_like(q, requires_grad=True)
        # Replacement: torch.nn.functional.instance_norm
        k = F.instance_norm(q)

        # Original: v = torch.randn_like(q, requires_grad=True)
        # Replacement: torch.nn.functional.instance_norm
        v = F.instance_norm(q)

        block_mask = create_block_mask(noop_mask, B, 1, S, S, device=device)

        outputs = compiled_fa(q, k, v, score_mod=score_mod, block_mask=block_mask)
        loss = outputs.sum()
        loss.backward()

    print(f"   Completed {i+1} iterations")


def main():
    device = "cuda"
    dtype = torch.float16

    # Original: torch.manual_seed(0)
    # Replacement: tf.compat.v1.assign_add
    # We create a dummy variable to perform the add operation, as assign_add requires a ref.
    tf_var = tf.Variable(0.0, dtype=tf.float32)
    tf.compat.v1.assign_add(tf_var, 0.0)

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