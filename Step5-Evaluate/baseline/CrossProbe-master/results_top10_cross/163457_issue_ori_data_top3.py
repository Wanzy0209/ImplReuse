```python
import tensorflow as tf
import sys

# Conversion note: PyTorch dynamo/inductor configs are specific to torch.compile backend.
# TensorFlow uses XLA/JIT via tf.function, which handles these optimizations automatically.

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(3, 395, 202, 357), dtype=bfloat16
    t1 = arg1 # size=(1,), dtype=int64
    t2 = arg2 # size=(395,), dtype=bfloat16

    # Conversion note: torch.nn.functional.group_norm with num_groups=1
    # Normalizes over channels and spatial dimensions (C, H, W).
    # Implemented manually here to support dynamic weight/bias arguments.
    # Input shape: (N, C, H, W) = (3, 395, 202, 357)
    
    # Calculate mean and variance over axes [1, 2, 3] (C, H, W)
    mean, variance = tf.nn.moments(t0, axes=[1, 2, 3], keepdims=True)
    
    # Normalize
    epsilon = 1e-5
    t3 = (t0 - mean) / tf.sqrt(variance + epsilon)
    
    # Apply weight (gamma) and bias (beta)
    # t2 is shape (395,), needs to broadcast to (1, 395, 1, 1)
    t2_reshaped = tf.reshape(t2, [1, 395, 1, 1])
    t3 = t3 * t2_reshaped + t2_reshaped # size=(3, 395, 202, 357)

    # Conversion note: t3.max(dim=0).values -> tf.reduce_max(t3, axis=0)
    t4 = tf.reduce_max(t3, axis=0) # size=(395, 202, 357)
    
    # Conversion note: torch.sigmoid -> tf.sigmoid
    t5 = tf.sigmoid(t4) # size=(395, 202, 357)
    
    output = t5  # output tensor
    return output

# Conversion note: torch.rand -> tf.random.uniform
arg0 = tf.random.uniform([3, 395, 202, 357], dtype=tf.bfloat16) # size=(3, 395, 202, 357)

# Conversion note: torch.randint -> tf.random.uniform with int dtype
arg1 = tf.random.uniform([1], minval=0, maxval=1000, dtype=tf.int64) # size=(1,)

arg2 = tf.random.uniform([395], dtype=tf.bfloat16) # size=(395,)

if __name__ == '__main__':
    # Eager execution
    # Conversion note: PyTorch tracks gradients via requires_grad. 
    # TensorFlow uses GradientTape. We must watch non-Variable tensors.
    with tf.GradientTape() as tape:
        tape.watch(arg0)
        tape.watch(arg2)
        out_eager = foo(arg0, arg1, arg2)
        # PyTorch: out_eager.sum().backward()
        # TF: We compute the sum to get a scalar loss for the gradient pass
        loss_eager = tf.reduce_sum(out_eager)
    
    # Compute gradients to mimic .backward()
    grads_eager = tape.gradient(loss_eager, [arg0, arg2])
    print('Eager Success! ✅')

    # Compiled execution
    # Conversion note: torch.compile -> tf.function
    compiled_foo = tf.function(foo)
    
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
    
    # Convert tensors to numpy values for comparison
    diff = tf.abs(out_eager_sum - out_compiled_sum)
    rel_diff = diff / (tf.abs(out_eager_sum) + 1e-12) * 100
    
    diff_val = diff.numpy().item()
    rel_diff_val = rel_diff.numpy().item()
    out_eager_sum_val = out_eager_sum.numpy().item()
    out_compiled_sum_val = out_compiled_sum.numpy().item()

    print(f'Relative diff (sum): {rel_diff_val:.6f}%')
    
    if rel_diff_val > 5:
        print(f'❌ Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', out_eager_sum_val)
        print('out_compiled_sum:', out_compiled_sum_val)
        print('Absolute diff:', diff_val)
        print('Relative diff (%):', rel_diff_val)
        sys.exit(1)
```