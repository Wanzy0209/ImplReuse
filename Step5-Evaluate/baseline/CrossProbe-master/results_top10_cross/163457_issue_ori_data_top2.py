```python
import tensorflow as tf
import sys

# PyTorch specific configurations are ignored in TensorFlow
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(3, 395, 202, 357), dtype=bfloat16
    t1 = arg1 # size=(1,), dtype=int64
    t2 = arg2 # size=(395,), dtype=bfloat16
    
    # Conversion: torch.nn.functional.group_norm(t0, 1, weight=t2, bias=t2)
    # GroupNorm with num_groups=1 normalizes over C, H, W (axes 1, 2, 3)
    # and applies affine transform (weight/bias) on the channel dimension.
    axes = [1, 2, 3]
    mean, variance = tf.nn.moments(t0, axes=axes, keepdims=True)
    epsilon = 1e-5
    t3 = (t0 - mean) * tf.math.rsqrt(variance + epsilon)
    
    # Reshape weight/bias (t2) to broadcast over (N, C, H, W)
    # t2 shape is (395,), needs to be (1, 395, 1, 1)
    t2_reshaped = tf.reshape(t2, [1, 395, 1, 1])
    t3 = t3 * t2_reshaped + t2_reshaped
    # size=(3, 395, 202, 357), dtype=bfloat16
    
    # Conversion: t3.max(dim=0).values
    t4 = tf.reduce_max(t3, axis=0) # size=(395, 202, 357), dtype=bfloat16
    
    # Conversion: torch.sigmoid(t4)
    t5 = tf.sigmoid(t4) # size=(395, 202, 357), dtype=bfloat16
    
    output = t5  # output tensor
    return output

# Conversion: torch.rand -> tf.random.uniform
# Conversion: requires_grad=True -> tf.Variable
arg0 = tf.Variable(tf.random.uniform([3, 395, 202, 357], dtype=tf.bfloat16)) # size=(3, 395, 202, 357), dtype=bfloat16
arg1 = tf.random.uniform([1], minval=0, maxval=1000, dtype=tf.int64) # size=(1,), dtype=int64
arg2 = tf.Variable(tf.random.uniform([395], dtype=tf.bfloat16)) # size=(395,), dtype=bfloat16

if __name__ == '__main__':
    # Eager execution
    with tf.GradientTape() as tape:
        out_eager = foo(arg0, arg1, arg2)
        loss_eager = tf.reduce_sum(out_eager)
    grads_eager = tape.gradient(loss_eager, [arg0, arg2])
    print('Eager Success! ✅')
    
    # Compiled execution (tf.function equivalent to torch.compile)
    compiled_foo = tf.function(foo)
    with tf.GradientTape() as tape:
        out_compiled = compiled_foo(arg0, arg1, arg2)
        loss_compiled = tf.reduce_sum(out_compiled)
    grads_compiled = tape.gradient(loss_compiled, [arg0, arg2])
    print('Compile Success! ✅')
    
    # Compare outputs (forward)
    out_eager_sum = tf.reduce_sum(out_eager)
    out_compiled_sum = tf.reduce_sum(out_compiled)
    diff = tf.abs(out_eager_sum - out_compiled_sum)
    rel_diff = diff / (tf.abs(out_eager_sum) + 1e-12) * 100
    print(f'Relative diff (sum): {rel_diff.numpy():.6f}%')
    
    if rel_diff > 5:
        print(f'❌ Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', out_eager_sum.numpy())
        print('out_compiled_sum:', out_compiled_sum.numpy())
        print('Absolute diff:', diff.numpy())
        print('Relative diff (%):', rel_diff.numpy())
        sys.exit(1)
```