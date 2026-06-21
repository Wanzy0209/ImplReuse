import torch
from functools import partial

# Helper function from the bug report
def _score_mode_fn_visibility(batch, head, q_idx, kv_idx, lower_bound, upper_bound):
    return (kv_idx >= lower_bound[q_idx]) & (kv_idx <= upper_bound[q_idx])

def create_attn_visibility(batch_size, seq_len):
    start = [x * seq_len for x in range(batch_size)]
    end = [x + (seq_len - 1) for x in start]
    attn_visibility = torch.tensor([start, end], dtype=torch.int32, device="cuda")
    return attn_visibility.view(2, -1)

def test_dynamo_cache_bug_create_block_mask():
    """
    Test case for Issue 164608: Dynamo cache bug create_block_mask.
    
    This test verifies that torch.compile(create_block_mask) correctly handles
    state reset between runs with different batch sizes. Without the fix,
    q_num_blocks contains garbage values from the previous run.
    
    We leverage torch.backends.cuda.is_flash_attention_available to ensure
    the test only runs in environments supporting the necessary CUDA features.
    """
    
    # Handle missing module for older PyTorch versions
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError:
        print("Skipping test: torch.nn.attention.flex_attention not available (requires PyTorch 2.4+).")
        return

    # Reuse the similar API to check if the environment supports the test
    if not torch.backends.cuda.is_flash_attention_available():
        print("Skipping test: Flash Attention not available.")
        return

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.manual_seed(10007)

    seq_len = 1024
    
    # Compile the function once
    compiled_create_block_mask = torch.compile(create_block_mask)

    # STEP 1: Run with batch_size=1 to populate the cache
    batch_size_1 = 1
    attn_visibility_1 = create_attn_visibility(batch_size_1, seq_len)
    kv_seqlen_1 = batch_size_1 * seq_len
    
    mask_1 = compiled_create_block_mask(
        partial(
            _score_mode_fn_visibility,
            lower_bound=attn_visibility_1.view(2, -1)[0],
            upper_bound=attn_visibility_1.view(2, -1)[1],
        ),
        1,
        None,
        kv_seqlen_1,
        kv_seqlen_1,
        device="cuda",
    )
    print(f"Run 1 (BS={batch_size_1}): q_num_blocks={mask_1.q_num_blocks}")

    # STEP 2: Run with batch_size=2 to trigger the cache bug
    batch_size_2 = 2
    attn_visibility_2 = create_attn_visibility(batch_size_2, seq_len)
    kv_seqlen_2 = batch_size_2 * seq_len

    # Get the ground truth via eager execution
    eager_mask_2 = create_block_mask(
        partial(
            _score_mode_fn_visibility,
            lower_bound=attn_visibility_2.view(2, -1)[0],
            upper_bound=attn_visibility_2.view(2, -1)[1],
        ),
        1,
        None,
        kv_seqlen_2,
        kv_seqlen_2,
        device="cuda",
    )

    # Get the result from the compiled function (reusing the cache)
    compiled_mask_2 = compiled_create_block_mask(
        partial(
            _score_mode_fn_visibility,
            lower_bound=attn_visibility_2.view(2, -1)[0],
            upper_bound=attn_visibility_2.view(2, -1)[1],
        ),
        1,
        None,
        kv_seqlen_2,
        kv_seqlen_2,
        device="cuda",
    )
    print(f"Run 2 (BS={batch_size_2}): q_num_blocks={compiled_mask_2.q_num_blocks}")

    # STEP 3: Verify that the compiled result matches the eager result
    assert torch.equal(compiled_mask_2.q_num_blocks, eager_mask_2.q_num_blocks), (
        f"Cache bug detected: Compiled q_num_blocks {compiled_mask_2.q_num_blocks} "
        f"does not match eager q_num_blocks {eager_mask_2.q_num_blocks}"
    )
    
    assert torch.equal(compiled_mask_2.kv_num_blocks, eager_mask_2.kv_num_blocks), (
        f"Cache bug detected: Compiled kv_num_blocks {compiled_mask_2.kv_num_blocks} "
        f"does not match eager kv_num_blocks {eager_mask_2.kv_num_blocks}"
    )

if __name__ == "__main__":
    test_dynamo_cache_bug_create_block_mask()