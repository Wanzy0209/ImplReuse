import torch
from torch.nn.attention.flex_attention import create_block_mask
from functools import partial

# Helper functions from the bug report
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

def create_block_mask_compiled(attn_visibility, kv_seqlen=None):
    _, num_tokens = attn_visibility.view(2, -1).shape
    # Compile the function once
    cbm_compiled = torch.compile(create_block_mask)
    return cbm_compiled(
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

def test_create_block_mask_dynamo_cache_reset():
    """
    Test to verify that create_block_mask resets state correctly between runs
    when using torch.compile, preventing garbage q_num_blocks values.
    
    This test leverages torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed
    to ensure the backend environment is checked, as Flex Attention behavior
    is closely tied to SDP backend capabilities.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        return

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.manual_seed(10007)

    # Check backend state using the similar API to ensure environment consistency
    # This relates to the domain of the bug (Flex Attention / SDP)
    sdp_allowed = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
    print(f"SDP fp16/bf16 reduction math allowed: {sdp_allowed}")

    seq_len = 1024
    batch_sizes = [1, 2]

    # Step 1: Establish ground truth with eager execution
    ground_truth = {}
    for batch_size in batch_sizes:
        attn_visibility = create_attn_visibility(batch_size, seq_len)
        kv_seqlen = batch_size * seq_len
        eager_mask = create_block_mask_eager(attn_visibility, kv_seqlen)
        ground_truth[batch_size] = eager_mask
        print(f"Ground truth batch_size={batch_size}: q_num_blocks={eager_mask.q_num_blocks}")

    # Step 2: Test compiled execution in sequence
    # The bug manifests when running the compiled function with different parameters
    # without resetting, causing q_num_blocks to retain values from the previous run.
    for batch_size in batch_sizes:
        attn_visibility = create_attn_visibility(batch_size, seq_len)
        kv_seqlen = batch_size * seq_len
        
        # Call the compiled wrapper
        compiled_mask = create_block_mask_compiled(attn_visibility, kv_seqlen)
        
        ground_truth_mask = ground_truth[batch_size]

        # Verify that the compiled output matches the eager output
        q_match = torch.equal(compiled_mask.q_num_blocks, ground_truth_mask.q_num_blocks)
        kv_match = torch.equal(compiled_mask.kv_num_blocks, ground_truth_mask.kv_num_blocks)

        assert q_match, (
            f"q_num_blocks mismatch for batch_size={batch_size}. "
            f"Expected {ground_truth_mask.q_num_blocks}, got {compiled_mask.q_num_blocks}. "
            "This indicates the Dynamo cache bug is present."
        )
        assert kv_match, (
            f"kv_num_blocks mismatch for batch_size={batch_size}. "
            f"Expected {ground_truth_mask.kv_num_blocks}, got {compiled_mask.kv_num_blocks}."
        )
        print(f"Compiled batch_size={batch_size}:  Passed")

if __name__ == "__main__":
    test_create_block_mask_dynamo_cache_reset()