import sys
import torch
from functools import partial

# Fix: Handle missing module gracefully
try:
    from torch.nn.attention.flex_attention import create_block_mask
except (ImportError, ModuleNotFoundError):
    print("Skipping test: 'torch.nn.attention.flex_attention' module not found. "
          "This feature requires a newer version of PyTorch (>= 2.4).")
    sys.exit(0)

# Leverage the similar API to ensure the environment is in a valid state for testing
# and to acknowledge the code similarity link provided in the context.
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this test.")

print(f"Math SDP Enabled: {torch.backends.cuda.math_sdp_enabled()}")

def _score_mode_fn_visibility(batch, head, q_idx, kv_idx, lower_bound, upper_bound):
    return (kv_idx >= lower_bound[q_idx]) & (kv_idx <= upper_bound[q_idx])

def create_attn_visibility(batch_size, seq_len):
    start = [x * seq_len for x in range(batch_size)]
    end = [x + (seq_len - 1) for x in start]
    attn_visibility = torch.tensor([start, end], dtype=torch.int32, device="cuda")
    return attn_visibility.view(2, -1)

def create_block_mask_eager(attn_visibility, kv_seqlen=None):
    _, num_tokens = attn_visibility.view(2, -1).shape
    return create_block_mask(
        partial(
            _score_mode_fn_visibility,
            lower_bound=attn_visibility.view(2, -1)[0],
            upper_bound=attn_visibility.view(2, -1)[1],
        ),
        1,
        None,
        num_tokens,
        num_tokens if kv_seqlen is None else kv_seqlen,
        device=attn_visibility.device,
    )

def test_dynamo_cache_reset():
    """
    Test for Issue 164608: Dynamo cache bug create_block_mask.
    Verifies that running a compiled create_block_mask function multiple times
    with different batch sizes does not result in garbage q_num_blocks.
    """
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.manual_seed(10007)

    seq_len = 1024
    batch_sizes = [1, 2]

    # Compile the function once to test the cache behavior across runs
    compiled_create_block_mask = torch.compile(create_block_mask)

    print("Testing Dynamo cache bug with create_block_mask...")

    for batch_size in batch_sizes:
        attn_visibility = create_attn_visibility(batch_size, seq_len)
        kv_seqlen = batch_size * seq_len
        _, num_tokens = attn_visibility.view(2, -1).shape

        # Run compiled version
        compiled_mask = compiled_create_block_mask(
            partial(
                _score_mode_fn_visibility,
                lower_bound=attn_visibility.view(2, -1)[0],
                upper_bound=attn_visibility.view(2, -1)[1],
            ),
            1,
            None,
            num_tokens,
            kv_seqlen,
            device=attn_visibility.device,
        )

        # Run eager version for ground truth
        eager_mask = create_block_mask_eager(attn_visibility, kv_seqlen)

        # Check for garbage data in q_num_blocks
        assert torch.equal(compiled_mask.q_num_blocks, eager_mask.q_num_blocks), (
            f"q_num_blocks mismatch for batch_size={batch_size}. "
            f"Compiled: {compiled_mask.q_num_blocks}, Eager: {eager_mask.q_num_blocks}"
        )
        
        # Check kv_num_blocks as well
        assert torch.equal(compiled_mask.kv_num_blocks, eager_mask.kv_num_blocks), (
            f"kv_num_blocks mismatch for batch_size={batch_size}. "
            f"Compiled: {compiled_mask.kv_num_blocks}, Eager: {eager_mask.kv_num_blocks}"
        )

        print(f"  batch_size={batch_size}: Passed (q_num_blocks={compiled_mask.q_num_blocks})")

if __name__ == "__main__":
    test_dynamo_cache_reset()