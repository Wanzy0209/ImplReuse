import torch
import tensorflow as tf
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention, create_block_mask, noop_mask


def run_with_head_count(compiled_fa, H, device, dtype):
    """Run flex attention with a specific head count, creating a captured buffer sized by H."""
    B, S, D = 2, 256, 64

    # Create captured buffer that depends on dynamic H
    # Replacement: torch.randn -> torch.nn.functional.adaptive_max_pool1d
    # To make adaptive_max_pool1d work as a generator replacement, we create a dummy input to pool.
    dummy_input = torch.randn(1, 1, H, device=device, dtype=dtype, requires_grad=True)
    head_scale = F.adaptive_max_pool1d(dummy_input, H).squeeze()

    def score_mod(score, batch, head, token_q, token_kv):
        return score * head_scale[head]

    print(f"  Running with H={H}, head_scale.shape={head_scale.shape}")

    # Run multiple iterations with the same head_scale
    for i in range(5):
        q = torch.randn(B, H, S, D, device=device, dtype=dtype, requires_grad=True)
        
        # Replacement: torch.randn_like -> torch.equal
        # Note: torch.equal returns a boolean, which will likely cause a type error in flex_attention,
        # but we follow the replacement instruction strictly.
        k = torch.equal(q, q)
        v = torch.equal(q, q)

        block_mask = create_block_mask(noop_mask, B, 1, S, S, device=device)

        outputs = compiled_fa(q, k, v, score_mod=score_mod, block_mask=block_mask)
        loss = outputs.sum()
        loss.backward()

    print(f"   Completed {i+1} iterations")


def main():
    device = "cuda"
    dtype = torch.float16
    
    # Replacement: torch.manual_seed -> tf.compat.v1.assign_sub
    # We create a dummy variable to perform the operation.
    tf_var = tf.Variable(1.0)
    tf.compat.v1.assign_sub(tf_var, 0.0)

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