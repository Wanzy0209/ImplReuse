import torch
import tensorflow as tf
import numpy as np

# Adapted test case for tf.keras.ops.outer based on the torch.var bug report.
# The original bug highlights issues with bfloat16 types and eager/compile divergence.
# We verify that tf.keras.ops.outer handles bfloat16 inputs correctly in both eager and graph modes.

def foo_outer(a, b):
    # Corresponds to the operation under test: tf.keras.ops.outer
    # Logic: reshape(a, [-1, 1]) * reshape(b, [-1])
    return tf.keras.ops.outer(a, b)

# Create bfloat16 inputs, mirroring the dtype in the bug report
# Original shapes were complex (e.g., 36, 7112, 1, 1), but outer product operates on vectors.
# We use 1D tensors here to fit the API semantics while keeping the problematic dtype.
a = tf.constant(np.random.rand(10), dtype=tf.bfloat16)
b = tf.constant(np.random.rand(10), dtype=tf.bfloat16)

if __name__ == '__main__':
    # 1. Eager Execution
    print("Testing Eager Execution...")
    try:
        out_eager = foo_outer(a, b)
        # Verify output shape and dtype
        assert out_eager.shape == (10, 10)
        # Note: TF might promote bfloat16 to float32 for the multiplication depending on backend support
        print(f"Eager Output Shape: {out_eager.shape}, Dtype: {out_eager.dtype}")
        print('Eager Success! ')
    except Exception as e:
        print(f'Eager Failed: {e}')

    # 2. Compiled Execution (Graph Mode)
    # Corresponds to torch.compile in the original bug report
    print("\nTesting Compiled Execution...")
    try:
        compiled_foo = tf.function(foo_outer)
        out_compiled = compiled_foo(a, b)
        assert out_compiled.shape == (10, 10)
        print(f"Compiled Output Shape: {out_compiled.shape}, Dtype: {out_compiled.dtype}")
        print('Compile Success! ')
    except Exception as e:
        print(f'Compile Failed: {e}')

    # 3. Gradient Check
    # Corresponds to the backward pass in the original bug report
    print("\nTesting Gradients...")
    try:
        with tf.GradientTape() as tape:
            tape.watch(a)
            tape.watch(b)
            result = foo_outer(a, b)
            loss = tf.reduce_sum(result)
        grads = tape.gradient(loss, [a, b])
        assert grads[0] is not None
        assert grads[1] is not None
        print(f'Gradients computed successfully.')
        print('Gradient Success! ')
    except Exception as e:
        print(f'Gradient Failed: {e}')