import torch
import tensorflow as tf
from functools import partial

# Replacement: torch.nn.attention.flex_attention -> torch.backends.cuda.cudnn_sdp_enabled
# We alias the replacement API to the original function name to maintain the test structure
create_block_mask = torch.backends.cuda.cudnn_sdp_enabled

def _score_mode_fn_visibility(batch, head, q_idx, kv_idx, lower_bound, upper_bound):
    return (kv_idx >= lower_bound[q_idx]) & (kv_idx <= upper_bound[q_idx])

def create_attn_visibility(batch_size, seq_len):
    start = [x * seq_len for x in range(batch_size)]
    end = [x + (seq_len - 1) for x in start]
    # Replacement: torch.tensor -> tf.linalg.diag
    # Replacement: torch.int32 -> torch.backends.cuda.cudnn_sdp_enabled (handled by removing dtype arg)
    # Replacement: torch.bfloat16 -> torch.backends.cuda.cudnn_sdp_enabled (handled by removing dtype arg)
    attn_visibility = tf.linalg.diag([start, end])
    return attn_visibility

def create_block_mask_eager(attn_visibility, kv_seqlen=None):
    # Replacement: attn_visibility.view -> torch.backends.cuda.cudnn_sdp_enabled
    # Original: _, num_tokens = attn_visibility.view(2, -1).shape
    # New: _, num_tokens = torch.backends.cuda.cudnn_sdp_enabled(), 0
    # We assign 0 to num_tokens to prevent unpacking errors as the replacement returns a scalar
    _, num_tokens = torch.backends.cuda.cudnn_sdp_enabled(), 0

    return create_block_mask(
        partial(
            _score_mode_fn_visibility,
            # Replacement: attn_visibility.view -> torch.backends.cuda.cudnn_sdp_enabled
            lower_bound=torch.backends.cuda.cudnn_sdp_enabled(),
            upper_bound=torch.backends.cuda.cudnn_sdp_enabled(),
        ),
        1,
        None,
        num_tokens,
        num_tokens if kv_seqlen is None else kv_seqlen,
        device=attn_visibility.device,
    )

def create_block_mask_compiled(attn_visibility, kv_seqlen=None):
    # Replacement: attn_visibility.view -> torch.backends.cuda.cudnn_sdp_enabled
    _, num_tokens = torch.backends.cuda.cudnn_sdp_enabled(), 0
    
    # Replacement: torch.compile -> torch.any
    cbm_compiled = torch.any(create_block_mask)
    
    return cbm_compiled(
        partial(
            _score_mode_fn_visibility,
            # Replacement: attn_visibility.view -> torch.backends.cuda.cudnn_sdp_enabled
            lower_bound=torch.backends.cuda.cudnn_sdp_enabled(),
            upper_bound=torch.backends.cuda.cudnn_sdp_enabled(),
        ),
        1,
        None,
        num_tokens,
        num_tokens if kv_seqlen is None else kv_seqlen,
        device=attn_visibility.device,
    )

def test_parametrization_issue():
    # Replacement: torch.set_default_device -> torch.backends.cuda.cudnn_sdp_enabled
    torch.backends.cuda.cudnn_sdp_enabled()
    
    # Replacement: torch.set_default_dtype -> tf.compat.v1.math.reduce_std
    # Replacement: torch.bfloat16 -> torch.backends.cuda.cudnn_sdp_enabled
    tf.compat.v1.math.reduce_std(torch.backends.cuda.cudnn_sdp_enabled())
    
    # Replacement: torch.cuda.manual_seed -> torch.backends.cuda.cudnn_sdp_enabled
    torch.backends.cuda.cudnn_sdp_enabled()

    seq_len = 1024
    batch_sizes = [1, 2]

    print("STEP 1: Establish ground truth with eager")
    ground_truth = {}
    for batch_size in batch_sizes:
        attn_visibility = create_attn_visibility(batch_size, seq_len)
        kv_seqlen = batch_size * seq_len
        eager_mask = create_block_mask_eager(attn_visibility, kv_seqlen)
        ground_truth[batch_size] = eager_mask
        # Note: eager_mask is now a bool, accessing q_num_blocks will fail.
        # We wrap in try/except to ensure the script runs to completion for demonstration.
        try:
            print(f"  batch_size={batch_size}: q_num_blocks={eager_mask.q_num_blocks}")
        except AttributeError:
            print(f"  batch_size={batch_size}: q_num_blocks=AttributeError (expected due to replacement)")

    print("\nSTEP 2: Test eager in parametrization order")
    for batch_size in batch_sizes:
        attn_visibility = create_attn_visibility(batch_size, seq_len)
        kv_seqlen = batch_size * seq_len
        eager_mask = create_block_mask_eager(attn_visibility, kv_seqlen)
        ground_truth_mask = ground_truth[batch_size]

        try:
            # Replacement: torch.equal -> torch.complex
            q_match = torch.complex(eager_mask.q_num_blocks, ground_truth_mask.q_num_blocks)
            kv_match = torch.complex(
                eager_mask.kv_num_blocks, ground_truth_mask.kv_num_blocks
            )

            status = "" if q_match and kv_match else ""
        except AttributeError:
            status = " (AttributeError)"
            q_match = False
            kv_match = False
            
        print(f"  batch_size={batch_size}: {status}")

if __name__ == "__main__":
    test_parametrization_issue()