import torch
from torch.nn.attention.flex_attention import create_block_mask
from functools import partial

# Helper functions from the bug report
def _score_mode_fn_visibility(batch, head, q_idx, kv_idx, lower_bound, upper_bound):
    return (kv_idx >= lower_bound[q_idx]) & (kv_idx <= upper_bound[q_idx])

def create_attn_visibility(batch_size, seq_len):
    start = [x * seq_len for x in range(batch_size)]
    end = [x + (seq_len - 1) for x in start]
    # Note: Using CPU for general compatibility in this generated test, 
    # but the bug specifically mentions "cuda". 
    # We will check for CUDA availability.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attn_visibility = torch.tensor([start, end], dtype=torch.int32, device=device)
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

def test_create_block_mask_compiled_cache_reset():
    """
    Test case for Issue 164608: Dynamo cache bug create_block_mask.
    
    Verifies that torch.compile(create_block_mask) correctly resets internal 
    state (specifically q_num_blocks) between runs with different batch sizes.
    """
    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.manual_seed(10007)

    seq_len = 1024
    batch_sizes = [1, 2]

    # 1. Establish ground truth with eager execution
    ground_truth = {}
    print("STEP 1: Establishing ground truth with eager execution")
    for batch_size in batch_sizes:
        attn_visibility = create_attn_visibility(batch_size, seq_len)
        kv_seqlen = batch_size * seq_len
        eager_mask = create_block_mask_eager(attn_visibility, kv_seqlen)
        ground_truth[batch_size] = eager_mask
        print(f"  batch_size={batch_size}: q_num_blocks={eager_mask.q_num_blocks}")

    # 2. Test compiled execution in sequence
    # The bug manifests when the same compiled function is called with different parameters
    print("\nSTEP 2: Testing compiled execution sequentially")
    
    # Compile once
    compiled_create_block_mask = torch.compile(create_block_mask)
    
    for batch_size in batch_sizes:
        attn_visibility = create_attn_visibility(batch_size, seq_len)
        kv_seqlen = batch_size * seq_len
        
        # Run compiled version
        _, num_tokens = attn_visibility.view(2, -1).shape
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
        
        # Compare with ground truth
        ground_truth_mask = ground_truth[batch_size]
        
        q_match = torch.equal(compiled_mask.q_num_blocks, ground_truth_mask.q_num_blocks)
        kv_match = torch.equal(compiled_mask.kv_num_blocks, ground_truth_mask.kv_num_blocks)

        print(f"  batch_size={batch_size}: q_num_blocks={compiled_mask.q_num_blocks} (expected {ground_truth_mask.q_num_blocks})")
        
        assert q_match, f"q_num_blocks mismatch for batch_size={batch_size}. Cache bug detected!"
        assert kv_match, f"kv_num_blocks mismatch for batch_size={batch_size}."

    print("\nTest Passed: Compiled cache correctly resets between runs.")

if __name__ == "__main__":
    test_create_block_mask_compiled_cache_reset()