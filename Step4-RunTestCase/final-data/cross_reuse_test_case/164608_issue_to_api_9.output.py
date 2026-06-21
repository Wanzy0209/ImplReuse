import torch
from functools import partial
import unittest

# Try importing the specific module, handle if missing
try:
    from torch.nn.attention.flex_attention import create_block_mask
except (ModuleNotFoundError, ImportError):
    create_block_mask = None

# Helper functions from the issue reproduction
def _score_mode_fn_visibility(batch, head, q_idx, kv_idx, lower_bound, upper_bound):
    return (kv_idx >= lower_bound[q_idx]) & (kv_idx <= upper_bound[q_idx])

def create_attn_visibility(batch_size, seq_len):
    start = [x * seq_len for x in range(batch_size)]
    end = [x + (seq_len - 1) for x in start]
    attn_visibility = torch.tensor([start, end], dtype=torch.int32, device="cuda")
    return attn_visibility.view(2, -1)

class TestCreateBlockMaskDynamoCache(unittest.TestCase):
    def setUp(self):
        if create_block_mask is None:
            self.skipTest("torch.nn.attention.flex_attention not available. Requires PyTorch 2.4+ or nightly.")

    def test_compiled_create_block_mask_cache_reset(self):
        """
        Test that create_block_mask resets its cache correctly between runs
        when using torch.compile, preventing garbage q_num_blocks.
        Leverages torch.backends.cuda.mem_efficient_sdp_enabled to verify
        the backend environment state.
        """
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        # Leverage the similar API to check the backend state.
        # This ensures the test runs in a context where attention optimizations
        # are active, which is relevant for flex_attention.
        sdp_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
        print(f"Testing with mem_efficient_sdp_enabled: {sdp_enabled}")

        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        torch.cuda.manual_seed(10007)

        seq_len = 1024
        batch_sizes = [1, 2]

        # Compile the function once
        compiled_create_block_mask = torch.compile(create_block_mask)

        # Run the compiled function for different batch sizes
        # The bug manifests if the cache from batch_size=1 corrupts batch_size=2
        results = {}
        for batch_size in batch_sizes:
            attn_visibility = create_attn_visibility(batch_size, seq_len)
            kv_seqlen = batch_size * seq_len
            _, num_tokens = attn_visibility.view(2, -1).shape

            mask = compiled_create_block_mask(
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
            results[batch_size] = mask

        # Verify the second run (batch_size=2) against a fresh eager execution
        # to ensure no cache pollution occurred.
        target_batch_size = 2
        attn_visibility = create_attn_visibility(target_batch_size, seq_len)
        kv_seqlen = target_batch_size * seq_len
        _, num_tokens = attn_visibility.view(2, -1).shape

        eager_mask = create_block_mask(
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

        compiled_mask = results[target_batch_size]

        self.assertTrue(
            torch.equal(compiled_mask.q_num_blocks, eager_mask.q_num_blocks),
            "Compiled q_num_blocks does not match eager. Dynamo cache bug detected."
        )
        self.assertTrue(
            torch.equal(compiled_mask.kv_num_blocks, eager_mask.kv_num_blocks),
            "Compiled kv_num_blocks does not match eager. Dynamo cache bug detected."
        )

if __name__ == "__main__":
    unittest.main()