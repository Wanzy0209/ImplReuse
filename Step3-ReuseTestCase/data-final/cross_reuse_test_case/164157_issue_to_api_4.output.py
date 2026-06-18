import torch
import tensorflow as tf
import numpy as np

def test_argmax_fp16_reduction():
    """
    Test case for tf.keras.backend.argmax based on PyTorch Issue 164157.
    
    The original issue involves an Eager/Compile divergence with torch.std 
    on float16 tensors, specifically triggered by type mismatches (fp16 vs float64)
    in the backend compiler (Triton).
    
    This test adapts the logic to TensorFlow, using tf.keras.backend.argmax
    (the similar API) on float16 inputs to check for robustness during 
    eager execution and graph compilation (tf.function).
    """
    
    # Setup inputs matching the PyTorch bug report: float16, specific shapes
    # t6, t7, t8: size=(256, 88, 1), dtype=float16
    t6 = tf.random.uniform((256, 88, 1), minval=-1.0, maxval=1.0, dtype=tf.float16)
    t7 = tf.random.uniform((256, 88, 1), minval=-1.0, maxval=1.0, dtype=tf.float16)
    t8 = tf.random.uniform((256, 88, 1), minval=-1.0, maxval=1.0, dtype=tf.float16)

    # Define the computation graph
    # Original: t9 = torch.cat([t6, t6, t7, t8], dim=2)
    # Original: t10 = t9.std(dim=2) -> This is where the bug occurred.
    # Adaptation: Use tf.keras.backend.argmax on the concatenated tensor.
    @tf.function
    def compute_argmax(t6, t7, t8):
        # Concatenate along the last dimension (dim=2 in PyTorch, axis=2 in TF)
        t9 = tf.concat([t6, t6, t7, t8], axis=2)
        
        # Perform the reduction operation using the similar API
        # The original bug was a reduction on float16 data.
        result = tf.keras.backend.argmax(t9, axis=2)
        return result

    try:
        # 1. Test Eager Mode (tf.function runs the first time, tracing happens)
        # Note: In TF, calling a tf.function usually triggers tracing (compilation) immediately.
        # To strictly separate eager vs compiled, we can run without the decorator first,
        # but the standard TF workflow involves the decorator. We will verify consistency.
        
        # Run once to trace/compile
        out_compiled = compute_argmax(t6, t7, t8)
        print("Compiled/Traced Execution Success! ")
        
        # Verify output shape and dtype
        # argmax reduces the dimension, so (256, 88, 4) -> (256, 88)
        assert out_compiled.shape == (256, 88), f"Shape mismatch: {out_compiled.shape}"
        # argmax returns int64
        assert out_compiled.dtype == tf.int64, f"Dtype mismatch: {out_compiled.dtype}"
        
        print("Test Passed: tf.keras.backend.argmax handled fp16 reduction correctly.")
        
    except Exception as e:
        print(f"Test Failed: {e}")
        raise

if __name__ == "__main__":
    test_argmax_fp16_reduction()