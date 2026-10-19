```python
import tensorflow as tf
import sys
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(3, 395, 202, 357), stride=(239370, 79790, 202, 1), dtype=bfloat16, device=cuda
    t1 = arg1 # size=(1,), stride=(1,), dtype=int64, device=cuda
    t2 = arg2 # size=(395,), stride=(1,), dtype=bfloat16, device=cuda
    # Conversion: torch.nn.functional.group_norm -> tf.nn.group_normalization
    # num_groups=1, channels_axis=1, reduction_axes=[1, 2, 3]
    # weight -> scale, bias -> offset
    t3 = tf.nn.group_normalization(t0, groups=1, channels_axis=1, reduction_axes=[1, 2, 3], scale=t2, offset=t2) # size=(3, 395, 202, 357), stride=(28485030, 72114, 357, 1), dtype=bfloat16, device=cuda
    # Conversion: t3.max(dim=0).values -> tf.reduce_max(t3, axis=0)
    t4 = tf.reduce_max(t3, axis=0) # size=(395, 202, 357), stride=(72114, 357, 1), dtype=bfloat16, device=cuda
    # Conversion: torch.sigmoid -> tf.math.sigmoid
    t5 = tf.math.sigmoid(t4) # size=(395, 202, 357), stride=(72114, 357, 1), dtype=bfloat16, device=cuda
    output = t5  # output tensor
    return output

# Conversion: torch.rand -> tf.random.uniform
# Conversion: requires_grad=True -> tf.Variable
arg0 = tf.Variable(tf.random.uniform([3, 395, 202, 357], dtype=tf.bfloat16)) # size=(3, 395, 202, 357), stride=(239370, 79790, 202, 1), dtype=bfloat16, device=cuda
# Conversion: torch.randint -> tf.random.uniform with dtype int64
arg1 = tf.random.uniform([1], minval=0, maxval=1000, dtype=tf.int64) # size=(1,), stride=(1,), dtype=int64, device=cuda
arg2 = tf.Variable(tf.random.uniform([395], dtype=tf.bfloat16)) # size=(395,), stride=(1,), dtype=bfloat16, device=cuda

if __name__ == '__main__':
    # Eager execution
    with tf.GradientTape() as tape:
        out_eager = foo(arg0, arg1, arg2)
        # PyTorch's backward() is called on the sum of outputs
        loss = tf.reduce_sum(out_eager)
    grads = tape.gradient(loss, [arg0, arg2])
    print('Eager Success! ✅')

    # Compiled execution
    # Conversion: torch.compile -> tf.function
    compiled_foo = tf.function(foo)
    with tf.GradientTape() as tape:
        out_compiled = compiled_foo(arg0, arg1, arg2)
        loss = tf.reduce_sum(out_compiled)
    grads = tape.gradient(loss, [arg0, arg2])
    print('Compile Success! ✅')

    # Compare outputs (forward)
    out_eager_sum = tf.reduce_sum(out_eager)
    out_compiled_sum = tf.reduce_sum(out_compiled)
    # Conversion: .abs().item() -> tf.abs(...).numpy().item()
    diff = tf.abs(out_eager_sum - out_compiled_sum).numpy().item()
    rel_diff = diff / (tf.abs(out_eager_sum).numpy().item() + 1e-12) * 100
    print(f'Relative diff (sum): {rel_diff:.6f}%')
    if rel_diff > 5:
        print(f'❌ Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', out_eager_sum.numpy().item())
        print('out_compiled_sum:', out_compiled_sum.numpy().item())
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)
```