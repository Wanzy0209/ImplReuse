import torch
import sys

# Attempt to import TensorFlow, handling potential environment issues (e.g., GLIBCXX version mismatch)
try:
    import tensorflow as tf
except ImportError as e:
    print(f"Skipping test: TensorFlow import failed due to environment issues (e.g., GLIBCXX version).")
    print(f"Error details: {e}")
    sys.exit(0)

def test_exp_precision_with_xla():
    """
    Test case adapted from PyTorch Issue 161944.
    Verifies if XLA compilation (similar to torch.compile) 
    affects the precision of tf.math.exp compared to a float64 reference.
    """
    # Leverage the similar API: Check if TensorFlow was built with XLA support
    if not tf.test.is_built_with_xla():
        print("TensorFlow not built with XLA, skipping test.")
        return

    # Reproduce the original bug logic
    # Generate random input
    inp = tf.random.normal((8192,))

    # Standard execution (float32)
    out1 = tf.math.exp(inp)

    # Compiled execution using XLA (float32)
    @tf.function(jit_compile=True)
    def compiled_exp(x):
        return tf.math.exp(x)

    out2 = compiled_exp(inp)

    # High precision reference (float64)
    out3_high = tf.math.exp(tf.cast(inp, tf.float64))

    # Compare errors against the high precision reference
    diff_standard = tf.reduce_max(tf.abs(out3_high - tf.cast(out1, tf.float64)))
    diff_compiled = tf.reduce_max(tf.abs(out3_high - tf.cast(out2, tf.float64)))

    print(f"Max difference (Standard vs Float64): {diff_standard.numpy()}")
    print(f"Max difference (XLA vs Float64): {diff_compiled.numpy()}")

    # Note: In the original PyTorch bug, the compiled version (out2) 
    # showed significantly higher error due to fast math optimizations.
    # This test allows checking if TensorFlow's XLA exhibits similar behavior.

if __name__ == "__main__":
    test_exp_precision_with_xla()