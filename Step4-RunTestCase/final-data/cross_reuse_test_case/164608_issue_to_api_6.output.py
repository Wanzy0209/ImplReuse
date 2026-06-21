import torch
from functools import partial
import enum

# Handle the import gracefully for environments where torch.nn.attention is not available
try:
    from torch.nn.attention.flex_attention import create_block_mask
except (ModuleNotFoundError, ImportError):
    create_block_mask = None

# Leveraging the pattern from the similar API (tf.tpu.experimental.DeviceOrderMode)
# to define execution modes for the test, translating the concept of "Device Order Modes"
# to "Execution Modes" (Eager vs Compiled) to structure the test case.
class ExecutionMode(enum.IntEnum):
    """The mode of execution for the create_block_mask function."""
    EAGER = 0
    COMPILED = 1


def _score_mode_fn_visibility(batch, head, q_idx, kv_idx, lower_bound, upper_bound):
    return (kv_idx >= lower_bound[q_idx]) & (kv_idx <= upper_bound[q_idx])


def create_attn_visibility(batch_size, seq_len):
    start = [x * seq_len for x in range(batch_size)]
    end = [x + (seq_len - 1) for x in start]
    attn_visibility = torch.tensor([start, end], dtype=torch.int32, device="cuda")
    return attn_visibility.view(2, -1)


def run_create_block_mask(mode, attn_visibility, kv_seqlen, compiled_fn=None):
    _, num_tokens = attn_visibility.view(2, -1).shape
    
    mask_fn = partial(
        _score_mode_fn_visibility,
        lower_bound=attn_visibility.view(2, -1)[0],
        upper_bound=attn_visibility.view(2, -1)[1],
    )

    if mode == ExecutionMode.COMPILED:
        if compiled_fn is None:
            # Compile the function once to simulate the caching scenario
            compiled_fn = torch.compile(create_block_mask)
        return compiled_fn(mask_fn, 1, None, num_tokens, kv_seqlen, device=attn_visibility.device), compiled_fn
    else:
        return create_block_mask(mask_fn, 1, None, num_tokens, kv_seqlen, device=attn_visibility.device), None


def test_dynamo_cache_bug_with_modes():
    """
    Test case to reproduce the Dynamo cache bug where create_block_mask 
    returns garbage q_num_blocks when run multiple times with different 
    batch sizes in compiled mode.
    """
    if not torch.cuda.is_available():
        print("Skipping test: CUDA not available")
        return

    if create_block_mask is None:
        print("Skipping test: torch.nn.attention.flex_attention not available (requires newer PyTorch)")
        return

    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.manual_seed(10007)

    seq_len = 1024
    batch_sizes = [1, 2]
    
    # Store ground truth results from EAGER execution
    ground_truth = {}
    
    print("STEP 1: Establish ground truth using EAGER mode")
    for batch_size in batch_sizes:
        attn_visibility = create_attn_visibility(batch_size, seq_len)
        kv_seqlen = batch_size * seq_len
        
        mask, _ = run_create_block_mask(ExecutionMode.EAGER, attn_visibility, kv_seqlen)
        ground_truth[batch_size] = mask.q_num_blocks.clone()
        print(f"  [EAGER] batch_size={batch_size}: q_num_blocks={mask.q_num_blocks.item()}")

    print("\nSTEP 2: Test COMPILED mode with sequential runs (Cache Check)")
    # We initialize the compiled function once to force caching behavior
    compiled_fn = None
    
    for i, batch_size in enumerate(batch_sizes):
        attn_visibility = create_attn_visibility(batch_size, seq_len)
        kv_seqlen = batch_size * seq_len
        
        # Pass the compiled_fn to ensure we reuse the same compiled graph
        mask, compiled_fn = run_create_block_mask(
            ExecutionMode.COMPILED, attn_visibility, kv_seqlen, compiled_fn
        )
        
        expected_q_blocks = ground_truth[batch_size]
        actual_q_blocks = mask.q_num_blocks
        
        # The bug manifests here: the second run (batch_size=2) might return 
        # the q_num_blocks from the first run (batch_size=1) or garbage.
        match = torch.equal(actual_q_blocks, expected_q_blocks)
        
        status = " PASS" if match else " FAIL"
        print(f"  [COMPILED Run {i+1}] batch_size={batch_size}: "
              f"q_num_blocks={actual_q_blocks.item()} (expected {expected_q_blocks.item()}) {status}")
        
        assert match, (
            f"Dynamo cache bug detected: q_num_blocks mismatch for batch_size={batch_size}. "
            f"Expected {expected_q_blocks.item()}, got {actual_q_blocks.item()}."
        )

if __name__ == "__main__":
    test_dynamo_cache_bug_with_modes()