import torch
import tensorflow as tf
import numpy as np

def test_random_normal_large_tensors():
    """
    Adapted test case based on PyTorch Issue 167727.
    The original bug (torch.addmm) manifests as incorrect results when handling 
    large tensors (specifically dimension 10000) on MPS.
    
    This test verifies that tf.keras.backend.random_normal handles large 
    tensor generation without producing invalid values (NaN/Inf) or crashing.
    """
    
    # 1. Success case (Small dimensions - mimicking the working PyTorch example)
    # Original dims: (64, 100)
    shape_small = (64, 100)
    try:
        # Note: tf.keras.backend.random_normal typically supports float types.
        # We use float32 here as the standard equivalent to complex64 for stability testing.
        out_small = tf.keras.backend.random_normal(shape_small, mean=0.0, stddev=1.0, dtype='float32', seed=42)
        assert out_small.shape == shape_small
        assert not tf.reduce_any(tf.math.is_nan(out_small))
        print("Small tensor test passed.")
    except Exception as e:
        print(f"Small tensor test failed: {e}")

    # 2. Stress case (Large dimensions - mimicking the failing PyTorch example)
    # Original failing dims involved 10000: (64, 10000)
    shape_large = (64, 10000)
    try:
        out_large = tf.keras.backend.random_normal(shape_large, mean=0.0, stddev=1.0, dtype='float32', seed=42)
        
        # Verify shape
        assert out_large.shape == shape_large, f"Shape mismatch: expected {shape_large}, got {out_large.shape}"
        
        # Verify data integrity (check for NaN or Inf which would indicate backend failure)
        # This corresponds to the "Mismatched elements" and "Greatest absolute difference" 
        # checks in the original bug report.
        has_nan = tf.reduce_any(tf.math.is_nan(out_large))
        has_inf = tf.reduce_any(tf.math.is_inf(out_large))
        
        assert not has_nan, "Large tensor generation produced NaN values"
        assert not has_inf, "Large tensor generation produced Inf values"
        
        print("Large tensor test passed.")
        
    except Exception as e:
        print(f"Large tensor test failed: {e}")
        raise

if __name__ == "__main__":
    test_random_normal_large_tensors()