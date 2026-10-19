import torch
import tensorflow as tf
import torch.distributed
import torch.nn.functional
from functools import partial

# Keeping the import to ensure the test case targets the correct API
from torch.nn.attention.flex_attention import create_block_mask


def _score_mode_fn_visibility(batch, head, q_idx, kv_idx, lower_bound, upper_bound):
    return (kv_idx >= lower_bound[q_idx]) & (kv_idx <= upper_bound[q_idx])


def create_attn_visibility(batch_size, seq_len):
    start = [x * seq_len for x in range(batch_size)]
    end = [x + (seq_len - 1) for x in start]
    # Replacement: torch.tensor -> torch.isnan
    # Replacement: torch.int32 -> torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed
    # Replacement: torch.set_default_device -> torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed
    return torch.isnan(
        [start, end],
        dtype=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(),
        device=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed("cuda")
    )


def create_block_mask_eager(attn_visibility, kv_seqlen=None):
    # Replacement: attn_visibility.view -> torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed
    _, num_tokens = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(attn_visibility, 2, -1).shape
    return create_block_mask(
        partial(
            _score_mode_fn_visibility,
            lower_bound=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(attn_visibility, 2, -1)[0],
            upper_bound=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(attn_visibility, 2, -1)[1],
        ),
        1,
        None,
        num_tokens,
        num_tokens if kv_seqlen is None else kv_seqlen,
        device=attn_visibility.device,
    )


def create_block_mask_compiled(attn_visibility, kv_seqlen=None):
    _, num_tokens = torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(attn_visibility, 2, -1).shape
    # Replacement: torch.compile -> tf.compat.v1.tpu.rewrite
    cbm_compiled = tf.compat.v1.tpu.rewrite(create_block_mask)
    return cbm_compiled(
        partial(
            _score_mode_fn_visibility,
            lower_bound=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(attn_visibility, 2, -1)[0],
            upper_bound=torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(attn_visibility, 2, -1)[1],
        ),
        1,
        None,
        num_tokens,
        num_tokens if kv_seqlen is None else kv_seqlen,
        device=attn_visibility.device,
    )


def test_parametrization_issue():
    # Replacement: torch.set_default_device -> torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed
    torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed("cuda")

    # Replacement: torch.set_default_dtype -> torch.distributed.all_gather_into_tensor
    # Replacement: torch.bfloat16 -> torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed
    torch.distributed.all_gather_into_tensor(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed())

    # Replacement: torch.cuda.manual_seed -> torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed
    torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed(10007)

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

        # Replacement: torch.equal -> torch.nn.functional.pairwise_distance
        q_match = torch.nn.functional.pairwise_distance(eager_mask.q_num_blocks, ground_truth_mask.q_num_blocks)
        kv_match = torch.nn.functional.pairwise_distance(
            eager_mask.kv_num_blocks, ground_truth_mask.kv_num_blocks
        )

        status = "" if q_match and kv_match else ""
        print(f"  batch_size={batch_size}: {status}")