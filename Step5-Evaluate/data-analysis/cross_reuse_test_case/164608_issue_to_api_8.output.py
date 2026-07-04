import torch
from functools import partial

# Handle missing dependency gracefully
try:
    from torch.nn.attention.flex_attention import create_block_mask
except (ImportError, ModuleNotFoundError):
    create_block_mask = None

# Helper function to define visibility based on bounds
def _score_mode_fn_visibility(batch, head, q_idx, kv_idx, lower_bound, upper_bound):
    return (kv_idx >= lower_bound[q_idx]) & (kv_idx <= upper_bound[q_idx])

# Helper to create attention visibility tensors
def create_attn_visibility(batch_size, seq_len):
    start = [x * seq_len for x in range(batch_size)]
    end = [x + (seq_len - 1) for x in start]
    attn_visibility = torch.tensor([start, end], dtype=torch.int32, device="cuda")
    return attn_visibility.view(2, -1)

# Eager version of the mask creation
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

# Compiled version of the mask creation
# We cache the compiled function to simulate the persistent state that causes the bug
def create_block_mask_compiled(attn_visibility, kv_seqlen=None):
    _, num_tokens = attn_visibility.view(2, -1).shape
    
    # Compile once and reuse to trigger the potential caching bug
    if not hasattr(create_block_mask_compiled, "_compiled_fn"):
        create_block_mask_compiled._compiled_fn = torch.compile(create_block_mask)
        
    return create_block_mask_compiled._compiled_fn(
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

def test_create_block_mask_dynamo_cache():
    """
    Test to verify that create_block_mask does not produce garbage q_num_blocks
    when run multiple times with different parameters in a compiled context.
    """
    if create_block_mask is None:
        print("Skipping test: torch.nn.attention.flex_attention is not available in this environment.")
        return

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.manual_seed(10007)

    seq_len = 1024
    # Using different batch sizes to trigger the cache invalidation/bug
    batch_sizes = [1, 2, 4]

    print("STEP 1: Establish ground truth with eager execution")
    ground_truth = {}
    for batch_size in batch_sizes:
        attn_visibility = create_attn_visibility(batch_size, seq_len)
        kv_seqlen = batch_size * seq_len
        eager_mask = create_block_mask_eager(attn_visibility, kv_seqlen)
        ground_truth[batch_size] = eager_mask
        print(f"  batch_size={batch_size}: q_num_blocks={eager_mask.q_num_blocks}")

    print("\nSTEP 2: Test compiled execution with varying batch sizes")
    for batch_size in batch_sizes:
        attn_visibility = create_attn_visibility(batch_size, seq_len)
        kv_seqlen = batch_size * seq_len
        
        # Run the compiled version
        compiled_mask = create_block_mask_compiled(attn_visibility, kv_seqlen)
        ground_truth_mask = ground_truth[batch_size]

        # Check if the internal state (q_num_blocks) matches the ground truth
        q_match = torch.equal(compiled_mask.q_num_blocks, ground_truth_mask.q_num_blocks)
        kv_match = torch.equal(compiled_mask.kv_num_blocks, ground_truth_mask.kv_num_blocks)

        status = "" if q_match and kv_match else ""
        print(f"  batch_size={batch_size}: {status}")
        
        # Assert to ensure the bug is caught if it exists
        assert q_match, (
            f"q_num_blocks mismatch for batch_size={batch_size}. "
            f"Expected {ground_truth_mask.q_num_blocks}, got {compiled_mask.q_num_blocks}"
        )
        assert kv_match, (
            f"kv_num_blocks mismatch for batch_size={batch_size}. "
            f"Expected {ground_truth_mask.kv_num_blocks}, got {compiled_mask.kv_num_blocks}"
        )

if __name__ == "__main__":
    test_create_block_mask_dynamo_cache()