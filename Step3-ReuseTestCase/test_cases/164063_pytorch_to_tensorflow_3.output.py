import torch
import tensorflow as tf
import numpy as np

def run_test():
    # The original bug involves bfloat16 tensors and a specific shape.
    # We adapt the test to use tf.experimental.numpy.nextafter with these characteristics.
    # Original t1 shape: (28, 24, 3, 127), dtype: bfloat16
    
    shape = (28, 24, 3, 127)
    dtype = tf.bfloat16

    # Create inputs mimicking the original tensor properties
    # We use random values to ensure the operation is actually performed
    x1 = tf.random.uniform(shape, minval=-1.0, maxval=1.0, dtype=dtype)
    x2 = tf.random.uniform(shape, minval=-1.0, maxval=1.0, dtype=dtype)

    # 1. Test Eager Execution
    print("Testing Eager Execution...")
    try:
        # Call the target API directly
        out_eager = tf.experimental.numpy.nextafter(x1, x2)
        print(f"Eager Success!  Output shape: {out_eager.shape}, dtype: {out_eager.dtype}")
    except Exception as e:
        print(f"Eager Failed: {e}")

    # 2. Test Compiled Execution
    # The original bug was a divergence between Eager and Compile (torch.compile).
    # We use tf.function with jit_compile=True to simulate the compilation path.
    print("\nTesting Compiled Execution...")
    
    @tf.function(jit_compile=True)
    def compiled_nextafter(a, b):
        return tf.experimental.numpy.nextafter(a, b)

    try:
        out_compiled = compiled_nextafter(x1, x2)
        print(f"Compile Success!  Output shape: {out_compiled.shape}, dtype: {out_compiled.dtype}")
    except Exception as e:
        print(f"Compile Failed: {e}")

if __name__ == '__main__':
    run_test()