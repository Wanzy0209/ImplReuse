import torch
import tensorflow as tf
import numpy as np
import sys

def foo(arg0, arg1):
    """
    Adapted test case for tf.math.nextafter based on the torch.var bug report.
    The core logic involves reshaping a bfloat16 tensor and applying a math operation
    to check for type handling issues in compiled mode.
    """
    # t0 = arg0
    # Original PyTorch code reshaped (36, 7112, 1, 1) to (28, 24, 3, 127).
    # Note: The element counts in the original bug report (256032 vs 257472) 
    # do not match exactly. Here we ensure the input size matches the target reshape
    # to make the TensorFlow test runnable.
    target_shape = (28, 24, 3, 127)
    
    # t1 = t0.reshape(...)
    t1 = tf.reshape(arg0, target_shape)
    
    # Prepare second argument for nextafter
    t1_b = tf.reshape(arg1, target_shape)
    
    # Original API: t2 = t1.var(dim=2)
    # Similar API: tf.math.nextafter
    # We apply nextafter element-wise. The bug context is handling bfloat16 
    # during compilation.
    t2 = tf.math.nextafter(t1, t1_b)
    
    # To mimic the graph complexity, we perform a reduction (like the original var)
    # or simply return the result to verify the op execution.
    # Here we return the result to ensure the kernel compiled correctly.
    return t2

# Generate inputs
# We need enough elements for the reshape (28 * 24 * 3 * 127 = 257472)
input_size = 28 * 24 * 3 * 127

# Create bfloat16 tensors, matching the dtype involved in the original bug
arg0 = tf.random.uniform([input_size], minval=-1.0, maxval=1.0, dtype=tf.bfloat16)
arg1 = tf.random.uniform([input_size], minval=-1.0, maxval=1.0, dtype=tf.bfloat16)

if __name__ == '__main__':
    print("Running Eager Execution...")
    try:
        out_eager = foo(arg0, arg1)
        # Verify output dtype is preserved (bfloat16)
        assert out_eager.dtype == tf.bfloat16, f"Expected bfloat16, got {out_eager.dtype}"
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed: {e}')
        sys.exit(1)

    print("\nRunning Compiled Execution (tf.function)...")
    # Using jit_compile=True to mimic torch.compile(fullgraph=True)
    compiled_foo = tf.function(foo, jit_compile=True)
    
    try:
        out_compiled = compiled_foo(arg0, arg1)
        # Verify output dtype is preserved in compiled mode
        assert out_compiled.dtype == tf.bfloat16, f"Expected bfloat16, got {out_compiled.dtype}"
        print('Compile Success! ')
    except Exception as e:
        print(f'Compile Failed: {e}')
        import traceback
        traceback.print_exc()
        sys.exit(1)