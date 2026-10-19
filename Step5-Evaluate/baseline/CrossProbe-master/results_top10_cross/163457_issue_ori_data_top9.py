```python
import tensorflow as tf
import sys

# PyTorch specific configurations for dynamo/inductor have no direct equivalent in TensorFlow.
# TensorFlow uses XLA (tf.function(jit_compile=True)) for compilation.
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(3, 395, 202, 357), stride=(239370, 79790, 202, 1), dtype=bfloat16, device=cuda
    t1 = arg1 # size=(1,), stride=(1,), dtype=int64, device=cuda
    t2 = arg2 # size=(395,), stride=(1,), dtype=bfloat16, device=cuda

    # Conversion: torch.nn.functional.group_norm -> tf.nn.group_normalization
    # PyTorch groups=1 normalizes over C, H, W.
    # TF channels_axis=1 (C), reduction_axes=[2, 3] (H, W).
    # weight/bias map to scale/offset.
    t3 = tf.nn.group_normalization(
        t0,
        groups=1,
        channels_axis=1,
        reduction_axes=[2, 3],
        scale=t2,
        offset=t2
    ) # size=(3, 395, 202, 357), stride=(28485030, 72114, 357, 1), dtype=bfloat16, device=cuda

    # Conversion: t3.max(dim=0).values -> tf.reduce_max(t3, axis=0)
    t4 = tf.reduce_max(t3, axis=0) # size=(395, 202, 357), stride=(72114, 357, 1), dtype=bfloat16, device=cuda

    # Conversion: torch.sigmoid -> tf.math.sigmoid
    t5 = tf.math.sigmoid(t4) # size=(395, 202, 357), stride=(72114, 357, 1), dtype=bfloat16, device=cuda

    output = t5  # output tensor
    return output

# Conversion: torch.rand -> tf.random.uniform
# Conversion: requires_grad=True -> wrap in tf.Variable
# Conversion: device='cuda' -> TF handles device placement automatically or via context
arg0 = tf.Variable(tf.random.uniform([3, 395, 202, 357], dtype=tf.bfloat16)) # size=(3, 395, 202, 357), stride=(239370, 79790, 202, 1), dtype=bfloat16, device=cuda
arg1 = tf.random.uniform([1], minval=0, maxval=1000, dtype=tf.int64) # size=(1,), stride=(1,), dtype=int64, device=cuda
arg2 = tf.Variable(tf.random.uniform([395], dtype=tf.bfloat16)) # size=(395,), stride=(1,), dtype=bfloat16, device=cuda

if __name__ == '__main__':
    # Eager execution with GradientTape
    with tf.GradientTape() as tape:
        out_eager = foo(arg0, arg1, arg2)
    # PyTorch backward() computes gradients for all inputs with requires_grad=True.
    # TF GradientTape computes gradients for watched variables (arg0, arg2).
    grads_eager = tape.gradient(out_eager, [arg0, arg2])
    print('Eager Success! ✅')

    # Compiled execution using tf.function (XLA)
    # PyTorch: torch.compile -> TensorFlow: tf.function(jit_compile=True)
    compiled_foo = tf.function(foo, jit_compile=True)

    with tf.GradientTape() as tape:
        out_compiled = compiled_foo(arg0, arg1, arg2)
    grads_compiled = tape.gradient(out_compiled, [arg0, arg2])
    print('Compile Success! ✅')

    # Compare outputs (forward)
    out_eager_sum = tf.reduce_sum(out_eager)
    out_compiled_sum = tf.reduce_sum(out_compiled)

    # Convert tensors to numpy values for comparison
    out_eager_sum_val = out_eager_sum.numpy().item()
    out_compiled_sum_val = out_compiled_sum.numpy().item()

    diff = abs(out_eager_sum_val - out_compiled_sum_val)
    rel_diff = diff / (abs(out_eager_sum_val) + 1e-12) * 100
    print(f'Relative diff (sum): {rel_diff:.6f}%')

    if rel_diff > 5:
        print(f'❌ Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', out_eager_sum_val)
        print('out_compiled_sum:', out_compiled_sum_val)
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)
```