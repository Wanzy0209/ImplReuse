import torch
import tensorflow as tf

# Adapted from PyTorch Issue 165861
# Original Bug: torch.nn.functional.pad with mode="reflect" crashes on CUDA 
# when a batch dimension is larger than uint16 max value (2**16).
# 
# This test verifies if the similar API, tf.clip_by_norm, handles tensors 
# with dimensions larger than 2**16 without crashing or producing incorrect results.

def test_clip_by_norm_large_dimension():
    # Create a tensor where the first dimension is exactly 2**16 (65536)
    # This matches the failing condition in the original bug report.
    # Note: Memory footprint is small, only the dimension index is large.
    x = tf.random.normal((2**16, 2), dtype=tf.float32)

    # Execute the target API
    # We check if the operation completes without error on this specific shape.
    clip_norm = 1.0
    result = tf.clip_by_norm(x, clip_norm=clip_norm)

    # Verify shape preservation
    assert result.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {result.shape}"

    # Verify functional correctness
    # clip_by_norm with axes=None (default) calculates the global L2-norm.
    # The L2-norm of the result should be less than or equal to clip_norm.
    l2_norm = tf.norm(result, ord='euclidean')
    assert l2_norm <= clip_norm + 1e-5, f"L2-norm {l2_norm} exceeds clip_norm {clip_norm}"

    print("Test passed: tf.clip_by_norm handles dimensions > 2**16 correctly.")

if __name__ == "__main__":
    test_clip_by_norm_large_dimension()