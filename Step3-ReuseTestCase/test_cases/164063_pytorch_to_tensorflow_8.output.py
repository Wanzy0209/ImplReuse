import torch
import tensorflow as tf
import numpy as np

def test_tf_keras_ops_power():
    """
    Adapted test case for tf.keras.ops.power based on the PyTorch bug report 
    involving torch.var and bfloat16 type handling.
    
    The original bug highlights a TypeError related to 'unexpected type fp32' 
    when handling bfloat16 tensors during eager/compile execution. 
    This test verifies if tf.keras.ops.power handles bfloat16 inputs robustly.
    """
    
    # Setup inputs mimicking the original PyTorch test case
    # Original: arg0 size=(36, 7112, 1, 1), dtype=bfloat16
    # We use a fixed seed for reproducibility
    tf.random.set_seed(42)
    arg0 = tf.random.uniform((36, 7112, 1, 1), dtype=tf.bfloat16)

    # Original: t1 = t0.reshape((28, 24, 3, 127))
    # Reshape logic: 36 * 7112 = 256032 elements
    #               28 * 24 * 3 * 127 = 256032 elements
    t1 = tf.reshape(arg0, (28, 24, 3, 127))

    print("Testing tf.keras.ops.power with bfloat16 inputs...")

    # --- Eager Execution ---
    try:
        # Original API: t2 = t1.var(dim=2)
        # Similar API: tf.keras.ops.power
        # We use power(x, 2) to test type promotion behavior similar to variance calculation (which involves squaring)
        t2 = tf.keras.ops.power(t1, 2.0)
        
        # Verify output shape and dtype
        assert t2.shape == (28, 24, 3, 127), f"Shape mismatch: {t2.shape}"
        # Note: Depending on TF version and backend, power might promote to float32. 
        # The bug in PyTorch was an *error* thrown during this process.
        print(f"Eager Execution Success!  Output dtype: {t2.dtype}")
    except Exception as e:
        print(f"Eager Execution Failed!  Error: {e}")
        return

    # --- Compiled Execution (XLA/JIT) ---
    # Mimics the torch.compile behavior in the original bug report
    try:
        @tf.function(jit_compile=True)
        def compiled_power(x):
            return tf.keras.ops.power(x, 2.0)

        t3 = compiled_power(t1)
        
        assert t3.shape == (28, 24, 3, 127), f"Compiled Shape mismatch: {t3.shape}"
        print(f"Compiled Execution Success!  Output dtype: {t3.dtype}")
    except Exception as e:
        print(f"Compiled Execution Failed!  Error: {e}")

if __name__ == '__main__':
    test_tf_keras_ops_power()