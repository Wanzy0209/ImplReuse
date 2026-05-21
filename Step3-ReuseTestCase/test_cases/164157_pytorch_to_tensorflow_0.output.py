import torch
import tensorflow as tf
import numpy as np

def test_inner_divergence():
    """
    Adapted test case for tf.experimental.numpy.inner based on the PyTorch std bug.
    The original bug involved a type mismatch (fp16 vs float64) during compilation
    with torch._dynamo. This test verifies that tf.experimental.numpy.inner
    handles float16 inputs consistently between eager and compiled (XLA) modes.
    """
    
    # Replicating the shape context from the original bug.
    # Original t9 shape: (256, 88, 4), dtype=float16
    # Original t10 (output of std) shape: (256, 88)
    # tf.experimental.numpy.inner on (..., 4) and (..., 4) reduces the last dim, resulting in (..., 88).
    shape = (256, 88, 4)
    
    # Create float16 inputs to stress the type system, similar to the original bug.
    # We use float16 to check for potential precision emulation or casting issues.
    a = tf.random.normal(shape, dtype=tf.float16)
    b = tf.random.normal(shape, dtype=tf.float16)

    # 1. Eager Execution
    print("Running Eager execution...")
    try:
        out_eager = tf.experimental.numpy.inner(a, b)
        print(f"Eager Success! Shape: {out_eager.shape}, Dtype: {out_eager.dtype}")
    except Exception as e:
        print(f"Eager Failed: {e}")
        return

    # 2. Compiled Execution (mimicking torch.compile)
    # Using jit_compile=True to trigger XLA compilation, which is analogous to 
    # the Inductor/Triton compilation path in the original PyTorch bug.
    print("\nRunning Compiled execution...")
    @tf.function(jit_compile=True)
    def compiled_op(x, y):
        return tf.experimental.numpy.inner(x, y)

    try:
        out_compiled = compiled_op(a, b)
        print(f"Compile Success! Shape: {out_compiled.shape}, Dtype: {out_compiled.dtype}")

        # Verify consistency between eager and compiled modes
        # We use a tolerance suitable for float16 arithmetic
        if not np.allclose(out_eager.numpy(), out_compiled.numpy(), rtol=1e-3, atol=1e-3):
            print("ERROR: Divergence detected between Eager and Compiled results!")
            print(f"Max diff: {np.max(np.abs(out_eager.numpy() - out_compiled.numpy()))}")
        else:
            print("SUCCESS: Eager and Compiled results match.")
    except Exception as e:
        print(f"Compile Failed: {e}")

if __name__ == "__main__":
    test_inner_divergence()