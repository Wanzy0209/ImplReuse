import torch
import tensorflow as tf
import numpy as np

def test_ctc_beam_search_decoder_extreme_input():
    """
    Test case adapted from PyTorch Issue 162251.
    Original issue: Floating point exception in torch.nn.PixelShuffle
    caused by a very large upscale factor (545460846592) and complex dtype.
    
    This test applies the same logic (extreme integer argument) to
    tf.nn.ctc_beam_search_decoder via the 'beam_width' parameter.
    """
    # The large integer value from the original bug report
    # which caused an overflow/FPE in shape checking logic.
    extreme_beam_width = 545460846592

    # Minimal valid inputs for the API
    # Shape: (max_time, batch_size, num_classes)
    max_time = 5
    batch_size = 1
    num_classes = 3
    
    # Note: Original bug used complex32, but tf.nn.ctc_beam_search_decoder requires float.
    # We stick to the 'extreme parameter' aspect of the bug reproduction logic.
    inputs = tf.constant(np.random.randn(max_time, batch_size, num_classes), dtype=tf.float32)
    sequence_length = tf.constant([max_time], dtype=tf.int32)

    try:
        # Attempt to decode with the extreme beam width.
        # This may trigger ResourceExhaustedError (OOM), InvalidArgumentError,
        # or potentially a crash if internal arithmetic overflows similarly to the PyTorch bug.
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(
            inputs,
            sequence_length,
            beam_width=extreme_beam_width
        )
        # If it doesn't crash, we check if it returned something (unlikely)
        assert decoded is not None
    except (tf.errors.ResourceExhaustedError, tf.errors.InvalidArgumentError) as e:
        # Expected behavior for extreme inputs in a robust system
        print(f"Caught expected error for extreme input: {e}")
    except Exception as e:
        # Catching other potential crashes or unexpected behaviors
        print(f"Unexpected exception: {e}")

if __name__ == "__main__":
    test_ctc_beam_search_decoder_extreme_input()