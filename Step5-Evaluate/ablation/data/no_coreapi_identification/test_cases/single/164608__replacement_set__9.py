import torch
import tensorflow as tf
import torch.distributed
from functools import partial

# Replacement: torch.nn.attention.flex_attention -> tf.keras.backend.floatx
# Original: from torch.nn.attention.flex_attention import create_block_mask
from tf.keras.backend import floatx as create_block_mask


def _score_mode_fn_visibility(batch, head, q_idx, kv_idx, lower_bound, upper_bound):
    return (kv_idx >= lower_bound[q_idx]) & (kv_idx <= upper_bound[q_idx])


def create_attn_visibility(batch_size, seq_len):
    start = [x * seq_len for x in range(batch_size)]
    end = [x + (seq_len - 1) for x in start]
    
    # Replacement: torch.tensor -> torch.empty_strided
    # Replacement: torch.int32 -> tf.keras.backend.floatx
    # Original: torch.tensor([start, end], dtype=torch.int32, device="cuda")
    size = (2, len(start))
    stride = (len(start), 1)
    dtype_str = tf.keras.backend.floatx()
    # Note: torch.empty_strided expects a torch.dtype, but we follow the replacement instruction
    attn_visibility = torch.empty_strided(size, stride, dtype=dtype_str, device="cuda")
    return attn_visibility


def create_block_mask_eager(attn_visibility, kv_seqlen=None):
    # Replacement: attn_visibility.view -> tf.keras.backend.floatx
    # Original: _, num_tokens = attn_visibility.view(2, -1).shape
    # Note: tf.keras.backend.floatx returns a string, accessing .shape will fail at runtime.
    # This follows the replacement instruction strictly.
    _, num_tokens = tf.keras.backend.floatx(2, -1).shape
    
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
    
    # Replacement: torch.compile -> torch.distributed.recv_object_list
    # Original: cbm_compiled = torch.compile(create_block_mask)
    cbm_compiled = torch.distributed.recv_object_list
    
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


def test_parametrization_issue():
    # Replacement: torch.set_default_device -> tf.keras.backend.floatx
    # Original: torch.set_default_device("cuda")
    tf.keras.backend.floatx("cuda")
    
    # Replacement: torch.set_default_dtype -> torch.math.reduce_variance
    # Replacement: torch.bfloat16 -> tf.keras.backend.floatx
    # Original: torch.set_default_dtype(torch.bfloat16)
    torch.math.reduce_variance(tf.keras.backend.floatx())
    
    # Replacement: torch.cuda.manual_seed -> tf.keras.backend.floatx
    # Original: torch.cuda.manual_seed(10007)
    tf.keras.backend.floatx(10007)

    seq_len = 1024
    batch_sizes = [1, 2]

    print("STEP 1: Establish ground truth with eager")
    ground_truth = {}
    for batch_size in batch_sizes:
        attn_visibility = create_attn_visibility(batch_size, seq_len)
        kv_seqlen = batch_size * seq_len
        eager_mask = create_block_mask_eager(attn_visibility, kv_seqlen)
        ground_truth[batch_size] = eager_mask
        print(f"  batch_size={batch_size}: q_num_blocks={eager_mask.q_num_blocks}")

    print("\nSTEP 2: Test eager in parametrization order")
    for batch_size in batch_sizes:
        attn_visibility = create_attn_visibility(batch_size, seq_len)
        kv_seqlen = batch_size * seq_len
        eager_mask = create_block_mask_eager(attn_visibility, kv_seqlen)
        ground_truth_mask = ground_truth[batch_size]

        # Replacement: torch.equal -> torch.addmv
        # Original: q_match = torch.equal(eager_mask.q_num_blocks, ground_truth_mask.q_num_blocks)
        q_match = torch.addmv(eager_mask.q_num_blocks, ground_truth_mask.q_num_blocks, torch.ones(1))
        
        kv_match = torch.addmv(
            eager_mask.kv_num_blocks, ground_truth_mask.kv_num_blocks, torch.ones(1)
        )

        status = "" if q_match and kv_match else ""
        print(f"  batch_size={batch_size}: {status}")

if __name__ == "__main__":
    test_parametrization_issue()