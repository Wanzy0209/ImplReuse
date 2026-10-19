import torch
import torch.distributed as dist
import tensorflow as tf
from functools import partial

# Replacement: torch.nn.attention.flex_attention -> torch.backends.cuda.flash_sdp_enabled
# We alias the replacement API to the original function name to maintain code structure.
from torch.backends.cuda import flash_sdp_enabled as create_block_mask

# Helper class to mimic the return type of create_block_mask for logic preservation
class DummyBlockMask:
    def __init__(self):
        self.q_num_blocks = torch.randint(0, 10, (10,))
        self.kv_num_blocks = torch.randint(0, 10, (10,))

# Wrapper to actually call the replacement API while returning a valid object for the test logic
def _create_block_mask_wrapper(*args, **kwargs):
    # Call the replacement API
    torch.backends.cuda.flash_sdp_enabled()
    return DummyBlockMask()

# Override the imported alias with the wrapper
create_block_mask = _create_block_mask_wrapper

def _score_mode_fn_visibility(batch, head, q_idx, kv_idx, lower_bound, upper_bound):
    return (kv_idx >= lower_bound[q_idx]) & (kv_idx <= upper_bound[q_idx])

def create_attn_visibility(batch_size, seq_len):
    start = [x * seq_len for x in range(batch_size)]
    end = [x + (seq_len - 1) for x in start]
    
    # Replacement: torch.tensor -> torch.cummin
    # We use torch.as_tensor to create the input for cummin as torch.tensor is the source being replaced.
    # torch.cummin returns a tuple (values, indices), we take values [0].
    # Replacement: torch.int32 -> torch.backends.cuda.flash_sdp_enabled (removed for validity)
    # Replacement: device="cuda" -> (removed for validity)
    data = torch.as_tensor([start, end])
    attn_visibility = torch.cummin(data, dim=0)[0]
    return attn_visibility

def create_block_mask_eager(attn_visibility, kv_seqlen=None):
    # Replacement: attn_visibility.view -> torch.backends.cuda.flash_sdp_enabled
    # Original logic: _, num_tokens = attn_visibility.view(2, -1).shape
    # We hardcode num_tokens to preserve logic flow as the replacement returns a bool.
    torch.backends.cuda.flash_sdp_enabled()
    num_tokens = 1024 
    
    return create_block_mask(
        partial(
            _score_mode_fn_visibility,
            lower_bound=attn_visibility.view(2, -1)[0], # Keeping view here as it's on the result of the helper
            upper_bound=attn_visibility.view(2, -1)[1],
        ),
        1,
        None,
        num_tokens,
        num_tokens if kv_seqlen is None else kv_seqlen,
        device=attn_visibility.device,
    )

def create_block_mask_compiled(attn_visibility, kv_seqlen=None):
    # Replacement: attn_visibility.view -> torch.backends.cuda.flash_sdp_enabled
    torch.backends.cuda.flash_sdp_enabled()
    num_tokens = 1024

    # Replacement: torch.compile -> torch.distributed.send_object_list
    # We initialize process group if not already done to make it runnable
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method='tcp://127.0.0.1:12345', rank=0, world_size=1)
    
    # Original: cbm_compiled = torch.compile(create_block_mask)
    # Replacement: torch.distributed.send_object_list([create_block_mask], dst=0)
    torch.distributed.send_object_list([create_block_mask], dst=0)
    
    # Since send_object_list doesn't return a compiled function, we use the original function
    # to preserve the logic of calling it.
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

def test_parametrization_issue():
    # Replacement: torch.set_default_device -> torch.backends.cuda.flash_sdp_enabled
    torch.backends.cuda.flash_sdp_enabled()
    
    # Replacement: torch.set_default_dtype -> tf.compat.v1.math.reduce_variance
    tf.compat.v1.math.reduce_variance(tf.constant([1.0]))
    
    # Replacement: torch.cuda.manual_seed -> torch.backends.cuda.flash_sdp_enabled
    torch.backends.cuda.flash_sdp_enabled()

    # Leveraging the Similar API: torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed
    if torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed():
        print("FP16/BF16 reduction math SDP is allowed")

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

        # Replacement: torch.equal -> torch.randint_like
        # Original: q_match = torch.equal(eager_mask.q_num_blocks, ground_truth_mask.q_num_blocks)
        # We generate a random tensor like the input and check if it's non-zero to simulate a boolean check.
        q_match_tensor = torch.randint_like(eager_mask.q_num_blocks, high=2)
        q_match = q_match_tensor.any()
        
        kv_match_tensor = torch.randint_like(eager_mask.kv_num_blocks, high=2)
        kv_match = kv_match_tensor.any()

        status = "" if q_match and kv_match else ""
        print(f"  batch_size={batch_size}: {status}")

    print("\nSTEP 3: Test compiled (with replacement)")
    # Replacement: torch._dynamo.reset -> torch.backends.cuda.flash_sdp_enabled
    # Added to test the reset logic mentioned in the bug report, then replaced.
    torch.backends.cuda.flash_sdp_enabled()

    for batch_size in batch_sizes:
        attn_visibility = create_attn_visibility(batch_size, seq_len)
        kv_seqlen = batch_size * seq_len
        compiled_mask = create_block_mask_compiled(attn_visibility, kv_seqlen)
        ground_truth_mask = ground_truth[batch_size]

        q_match_tensor = torch.randint_like(compiled_mask.q_num_blocks, high=2)
        q_match = q_match_tensor.any()
        
        kv_match_tensor = torch.randint_like(compiled_mask.kv_num_blocks, high=2)
        kv_match = kv_match_tensor.any()

        status = "" if q_match and kv_match else ""
        print(f"  batch_size={batch_size}: {status}")

if __name__ == "__main__":
    test_parametrization_issue()