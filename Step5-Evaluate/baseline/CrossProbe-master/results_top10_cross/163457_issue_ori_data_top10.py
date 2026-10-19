```python
import tensorflow as tf
import sys

# PyTorch specific configurations (ignored in TensorFlow translation)
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(3, 395, 202, 357), dtype=bfloat16
    t1 = arg1 # size=(1,), dtype=int64
    t2 = arg2 # size=(395,), dtype=bfloat16
    
    # Conversion: torch.nn.functional.group_norm with num_groups=1
    # is equivalent to Layer Normalization over the non-batch dimensions (C, H, W)
    # followed by an affine transform using the provided weight and bias.
    # PyTorch group_norm default epsilon is 1e-5.
    mean, variance = tf.nn.moments(t0, axes=[1, 2, 3], keepdims=True)
    epsilon = 1e-5
    t3_normalized = (t0 - mean) / tf.sqrt(variance + epsilon)
    
    # Apply affine transformation: output = normalized * weight + bias
    # t2 is used for both weight and bias.
    # t2 shape is (395,), t3_normalized shape is (3, 395, 202, 357)
    # Broadcasting will handle the alignment.
    t3 = t2 * t3_normalized + t2 # size=(3, 395, 202, 357)
    
    t4 = tf.reduce_max(t3, axis=0) # size=(395, 202, 357)
    t5 = tf.sigmoid(t4) # size=(395, 202, 357)
    output = t5  # output tensor
    return output

# Conversion: torch.rand -> tf.random.uniform
# Conversion: requires_grad=True -> tf.Variable
arg0 = tf.Variable(tf.random.uniform((3, 395, 202, 357), dtype=tf.bfloat16)) # size=(3, 395, 202, 357), dtype=bfloat16
# Conversion: torch.randint -> tf.random.uniform (with int dtype)
arg1 = tf.random.uniform((1,), minval=0, maxval=1000, dtype=tf.int64) # size=(1,), dtype=int64
arg2 = tf.Variable(tf.random.uniform((395,), dtype=tf.bfloat16)) # size=(395,), dtype=bfloat16

if __name__ == '__main__':
    # Eager execution
    with tf.GradientTape() as tape:
        out_eager = foo(arg0, arg1, arg2)
        loss = tf.reduce_sum(out_eager)
    grads = tape.gradient(loss, [arg0, arg2])
    print('Eager Success! ✅')
    
    # Compiled execution
    # Conversion: torch.compile -> tf.function with jit_compile=True
    compiled_foo = tf.function(foo, jit_compile=True)
    
    with tf.GradientTape() as tape:
        out_compiled = compiled_foo(arg0, arg1, arg2)
        loss = tf.reduce_sum(out_compiled)
    grads = tape.gradient(loss, [arg0, arg2])
    print('Compile Success! ✅')
    
    # Compare outputs (forward)
    out_eager_sum = tf.reduce_sum(out_eager)
    out_compiled_sum = tf.reduce_sum(out_compiled)
    
    # Conversion: .abs().item() -> tf.abs(...).numpy()
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