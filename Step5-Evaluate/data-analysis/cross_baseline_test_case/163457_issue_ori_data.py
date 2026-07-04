```python
import tensorflow as tf
import sys

# Note: PyTorch specific configs (torch._dynamo, torch._inductor) are omitted as they have no TF equivalent.

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(3, 395, 202, 357), dtype=bfloat16
    t1 = arg1 # size=(1,), dtype=int64
    t2 = arg2 # size=(395,), dtype=bfloat16
    
    # Conversion: torch.nn.functional.group_norm -> tf.nn.group_normalization
    # PyTorch uses NCHW format (Batch, Channels, Height, Width).
    # groups=1 implies normalizing over C, H, W.
    # channels_axis=1 corresponds to the 'C' in NCHW.
    # reduction_axes=[1, 2, 3] corresponds to C, H, W.
    t3 = tf.nn.group_normalization(
        t0, 
        groups=1, 
        channels_axis=1, 
        reduction_axes=[1, 2, 3], 
        scale=t2, # weight
        offset=t2, # bias
        epsilon=1e-5
    ) # size=(3, 395, 202, 357), dtype=bfloat16
    
    # Conversion: t3.max(dim=0).values -> tf.reduce_max(t3, axis=0)
    t4 = tf.reduce_max(t3, axis=0) # size=(395, 202, 357), dtype=bfloat16
    
    # Conversion: torch.sigmoid -> tf.math.sigmoid
    t5 = tf.math.sigmoid(t4) # size=(395, 202, 357), dtype=bfloat16
    
    output = t5  # output tensor
    return output

# Conversion: torch.rand -> tf.random.uniform
# Conversion: torch.bfloat16 -> tf.bfloat16
arg0 = tf.random.uniform([3, 395, 202, 357], minval=0.0, maxval=1.0, dtype=tf.bfloat16) # size=(3, 395, 202, 357), dtype=bfloat16

# Conversion: torch.randint -> tf.random.uniform (cast to int64)
arg1 = tf.cast(tf.random.uniform([1], minval=0, maxval=1000), dtype=tf.int64) # size=(1,), dtype=int64

arg2 = tf.random.uniform([395], minval=0.0, maxval=1.0, dtype=tf.bfloat16) # size=(395,), dtype=bfloat16

if __name__ == '__main__':
    # Eager execution
    with tf.GradientTape() as tape:
        # Conversion: requires_grad=True -> tape.watch
        tape.watch(arg0)
        tape.watch(arg2)
        out_eager = foo(arg0, arg1, arg2)
        # Conversion: .sum().backward() -> calculating loss and gradient
        loss_eager = tf.reduce_sum(out_eager)
    
    grads_eager = tape.gradient(loss_eager, [arg0, arg2])
    print('Eager Success! ✅')
    
    # Conversion: torch.compile -> tf.function with jit_compile=True
    compiled_foo = tf.function(foo, jit_compile=True)
    
    with tf.GradientTape() as tape:
        tape.watch(arg0)
        tape.watch(arg2)
        out_compiled = compiled_foo(arg0, arg1, arg2)
        loss_compiled = tf.reduce_sum(out_compiled)
        
    grads_compiled = tape.gradient(loss_compiled, [arg0, arg2])
    print('Compile Success! ✅')
    
    # Compare outputs (forward)
    out_eager_sum = tf.reduce_sum(out_eager)
    out_compiled_sum = tf.reduce_sum(out_compiled)
    
    # Conversion: .item() -> .numpy()
    diff = abs(out_eager_sum.numpy() - out_compiled_sum.numpy())
    rel_diff = diff / (abs(out_eager_sum.numpy()) + 1e-12) * 100
    print(f'Relative diff (sum): {rel_diff:.6f}%')
    
    if rel_diff > 5:
        print(f'❌ Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', out_eager_sum.numpy())
        print('out_compiled_sum:', out_compiled_sum.numpy())
        print('Absolute diff:', diff)
        print('Relative diff (%):', rel_diff)
        sys.exit(1)
```