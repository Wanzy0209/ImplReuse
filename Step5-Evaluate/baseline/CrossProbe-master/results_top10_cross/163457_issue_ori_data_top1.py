```python
import tensorflow as tf
import sys
# torch._dynamo.config.capture_scalar_outputs = True  # N/A for TensorFlow
# torch._dynamo.config.capture_dynamic_output_shape_ops = True  # N/A for TensorFlow
# torch._inductor.config.emulate_precision_casts = True  # N/A for TensorFlow

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(3, 395, 202, 357), dtype=bfloat16
    t1 = arg1 # size=(1,), dtype=int64
    t2 = arg2 # size=(395,), dtype=bfloat16
    # Conversion: torch.nn.functional.group_norm -> tf.raw_ops.GroupNorm
    # Note: PyTorch group_norm with num_groups=1 normalizes over C, H, W.
    # TF GroupNorm normalizes over all axes except batch and channels.
    # Input shape (N, C, H, W) -> (3, 395, 202, 357).
    # Channels is dim 1.
    # t2 is (395,), matching C.
    t3 = tf.raw_ops.GroupNorm(x=t0, groups=1, scale=t2, offset=t2, epsilon=1e-05) # size=(3, 395, 202, 357), dtype=bfloat16
    # Conversion: t3.max(dim=0).values -> tf.reduce_max(t3, axis=0)
    t4 = tf.reduce_max(t3, axis=0) # size=(395, 202, 357), dtype=bfloat16
    # Conversion: torch.sigmoid -> tf.keras.activations.sigmoid
    t5 = tf.keras.activations.sigmoid(t4) # size=(395, 202, 357), dtype=bfloat16
    output = t5  # output tensor
    return output

# Conversion: torch.rand -> tf.random.uniform
arg0 = tf.random.uniform([3, 395, 202, 357], dtype=tf.bfloat16) # size=(3, 395, 202, 357), dtype=bfloat16
# Conversion: torch.randint -> tf.random.uniform with int dtype
arg1 = tf.random.uniform([1], minval=0, maxval=1000, dtype=tf.int64) # size=(1,), dtype=int64
# Conversion: torch.rand -> tf.random.uniform
arg2 = tf.random.uniform([395], dtype=tf.bfloat16) # size=(395,), dtype=bfloat16

if __name__ == '__main__':
    # Eager execution with gradient tracking
    with tf.GradientTape() as tape:
        tape.watch(arg0)
        tape.watch(arg2)
        out_eager = foo(arg0, arg1, arg2)
    grads_eager = tape.gradient(out_eager, [arg0, arg2])
    print('Eager Success! ✅')
    
    # Conversion: torch.compile -> tf.function with jit_compile=True
    compiled_foo = tf.function(foo, jit_compile=True)
    with tf.GradientTape() as tape:
        tape.watch(arg0)
        tape.watch(arg2)
        out_compiled = compiled_foo(arg0, arg1, arg2)
    grads_compiled = tape.gradient(out_compiled, [arg0, arg2])
    print('Compile Success! ✅')
    
    # Compare outputs (forward)
    out_eager_sum = tf.reduce_sum(out_eager)
    out_compiled_sum = tf.reduce_sum(out_compiled)
    diff = tf.abs(out_eager_sum - out_compiled_sum).numpy()
    rel_diff = diff / (tf.abs(out_eager_sum).numpy() + 1e-12) * 100
    print(f'Relative diff (sum): {rel_diff:.6f}%')
    if rel_diff > 5:
        print(f'❌ Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', out_eager_sum.numpy())
        print('out_compiled_sum:', out_compiled_sum.numpy())
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)
```