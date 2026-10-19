```python
import tensorflow as tf
import sys

# Note: PyTorch specific configs (capture_scalar_outputs, etc.) are not directly 
# translatable to TensorFlow global configs, so they are omitted.

def unbiased_variance(x, axis=None, keepdims=False):
    # PyTorch var default is unbiased (N-1), TF reduce_variance is biased (N)
    # We adjust TF result to match PyTorch behavior
    variance = tf.math.reduce_variance(x, axis=axis, keepdims=keepdims)
    if axis is None:
        n = tf.cast(tf.size(x), tf.float32)
    else:
        # Handle axis being a tuple or int
        shape = tf.shape(x)
        n = 1.0
        if isinstance(axis, int):
            n = tf.cast(shape[axis], tf.float32)
        else:
            for ax in axis:
                n = n * tf.cast(shape[ax], tf.float32)
    
    # Avoid division by zero if n=1
    return variance * (n / (n - 1))

def foo(arg0, arg1, arg2, arg3, arg4):
    t0 = arg0 # size=(5, 4), dtype=bfloat16, device=cuda
    t1 = arg1 # size=(5, 1024), dtype=bfloat16, device=cuda
    t2 = arg2 # size=(1024, 4), dtype=bfloat16, device=cuda
    
    # Conversion: torch.addmm(t0, t1, t2) -> t0 + matmul(t1, t2)
    t3 = t0 + tf.linalg.matmul(t1, t2) # size=(5, 4), dtype=bfloat16, device=cuda
    
    # Conversion: t3.norm() -> Frobenius norm for matrices
    t4 = tf.norm(t3) # size=(), dtype=bfloat16, device=cuda
    
    t5 = arg3 # size=(3, 4, 5, 2), dtype=float32, device=cuda
    
    # Conversion: t5.var(dim=0) -> Variance along dim 0. PyTorch default is unbiased.
    t6 = unbiased_variance(t5, axis=0) # size=(4, 5, 2), dtype=float32, device=cuda
    
    # Conversion: t6.var() -> Unbiased variance of all elements
    t7 = unbiased_variance(t6) # size=(), dtype=float32, device=cuda
    
    t8 = arg4 # size=(), dtype=float32, device=cuda
    # Conversion: torch.nn.functional.relu -> tf.nn.relu
    t9 = tf.nn.relu(t8) # size=(), dtype=float32, device=cuda
    
    t10 = t7 + t4 + t9 # size=(), dtype=float32, device=cuda
    
    # Conversion: torch.pow -> tf.pow. Handles type promotion (bfloat16 ^ float32 -> float32)
    t11 = tf.pow(tf.pow(t4, t7), t10) # size=(), dtype=float32, device=cuda
    output = t11  # output tensor
    return output

# Initialize inputs
# Using tf.Variable to mimic requires_grad=True behavior for GradientTape
# Placing on GPU if available
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
with tf.device(device):
    # Conversion: torch.rand -> tf.random.uniform
    arg0 = tf.Variable(tf.random.uniform([5, 4], dtype=tf.bfloat16)) # size=(5, 4), dtype=bfloat16, device=cuda
    arg1 = tf.Variable(tf.random.uniform([5, 1024], dtype=tf.bfloat16)) # size=(5, 1024), dtype=bfloat16, device=cuda
    arg2 = tf.Variable(tf.random.uniform([1024, 4], dtype=tf.bfloat16)) # size=(1024, 4), dtype=bfloat16, device=cuda
    arg3 = tf.Variable(tf.random.uniform([3, 4, 5, 2], dtype=tf.float32)) # size=(3, 4, 5, 2), dtype=float32, device=cuda
    arg4 = tf.Variable(tf.random.uniform([], dtype=tf.float32)) # size=(), dtype=float32, device=cuda

if __name__ == '__main__':
    # Eager Execution
    with tf.GradientTape() as tape:
        out_eager = foo(arg0, arg1, arg2, arg3, arg4)
    # Compute gradients to mimic .backward()
    grads_eager = tape.gradient(out_eager, [arg0, arg1, arg2, arg3, arg4])
    print('Eager Success! ✅')
    
    # Compiled Execution
    # Conversion: torch.compile -> tf.function with jit_compile=True
    compiled_foo = tf.function(foo, jit_compile=True)
    
    with tf.GradientTape() as tape:
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4)
    grads_compiled = tape.gradient(out_compiled, [arg0, arg1, arg2, arg3, arg4])
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