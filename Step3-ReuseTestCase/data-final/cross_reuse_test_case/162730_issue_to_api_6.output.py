import torch
import tensorflow as tf
import numpy as np

def test_set_floatx_linear_consistency():
    """
    Test case adapted from PyTorch MPS F.Linear bug (Issue 162730).
    
    Original Bug Logic:
    - Compares results of linear operation with contiguous vs non-contiguous weights.
    - Expects numerical consistency (results should match).
    
    Adapted Logic for tf.keras.backend.set_floatx:
    - Compares results of linear operation under different float precision settings.
    - Verifies that switching global precision (set_floatx) produces consistent results
      when returning to the same precision (e.g., float32 -> float16 -> float32).
    - This ensures the global state change does not introduce side effects or garbage data,
      similar to how memory layout changes should not affect numerical results.
    """
    
    # Save original floatx to restore later
    original_floatx = tf.keras.backend.floatx()

    # Create test data (using float32 as baseline)
    # Dimensions mimic the original bug report: x(1, 3, 768), W(768, 768), bias(768)
    np.random.seed(42)
    x_np = np.random.randn(1, 3, 768).astype('float32')
    w_np = np.random.randn(768, 768).astype('float32')
    b_np = np.random.randn(768).astype('float32')

    # --- State 1: float32 ---
    tf.keras.backend.set_floatx('float32')
    # Using Dense layer as the equivalent of torch.nn.functional.linear
    layer_f32 = tf.keras.layers.Dense(768, use_bias=True)
    layer_f32.build((None, 768))
    layer_f32.set_weights([w_np, b_np])
    result1 = layer_f32(x_np)

    # --- State 2: float16 ---
    tf.keras.backend.set_floatx('float16')
    layer_f16 = tf.keras.layers.Dense(768, use_bias=True)
    layer_f16.build((None, 768))
    # Weights are cast to float16 by the layer based on the global policy
    layer_f16.set_weights([w_np, b_np])
    result2 = layer_f16(x_np)

    # --- State 3: float32 (Restored) ---
    tf.keras.backend.set_floatx('float32')
    layer_f32_restored = tf.keras.layers.Dense(768, use_bias=True)
    layer_f32_restored.build((None, 768))
    layer_f32_restored.set_weights([w_np, b_np])
    result3 = layer_f32_restored(x_np)

    # Assertions
    
    # 1. Check that float32 results are consistent.
    # This mirrors the original bug's check: "contiguous vs non-contiguous should match".
    # Here: "float32 state vs restored float32 state should match".
    print(f"Float32 vs Restored Float32 match: {np.allclose(result1.numpy(), result3.numpy())}")
    assert np.allclose(result1.numpy(), result3.numpy()), \
        "Results should match when returning to the same floatx setting (consistency check)."

    # 2. Check that float16 is actually different (sanity check for precision change).
    print(f"Float32 vs Float16 match: {np.allclose(result1.numpy(), result2.numpy())}")
    assert not np.allclose(result1.numpy(), result2.numpy()), \
        "Results should differ between float32 and float16."

    # 3. Verify dtypes are correct
    assert result1.dtype == tf.float32, "Result 1 should be float32"
    assert result2.dtype == tf.float16, "Result 2 should be float16"
    assert result3.dtype == tf.float32, "Result 3 should be float32"

    # Restore original state
    tf.keras.backend.set_floatx(original_floatx)
    print("Test passed.")

if __name__ == "__main__":
    test_set_floatx_linear_consistency()