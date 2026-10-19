```python
import tensorflow as tf
import sys

# Check for GPU availability to match 'cuda' device in source
gpus = tf.config.list_physical_devices('GPU')
device = '/GPU:0' if gpus else '/CPU:0'

def foo(arg0, arg1, arg2, arg3, arg4):
    t0 = arg0 # size=(5, 4), dtype=bfloat16, device=cuda
    t1 = arg1 # size=(5, 1024), dtype=bfloat16, device=cuda
    t2 = arg2 # size=(1024, 4), dtype=bfloat16, device=cuda
    
    # Conversion: torch.addmm(t0, t1, t2) -> t0 + matmul(t1, t2)
    t3 = t0 + tf.linalg.matmul(t1, t2) # size=(5, 4), dtype=bfloat16, device=cuda
    
    # Conversion: torch.norm() -> tf.norm()
    t4 = tf.norm(t3) # size=(), dtype=bfloat16, device=cuda
    
    t5 = arg3 # size=(3, 4, 5, 2), dtype=float32, device=cuda
    
    # Conversion: torch.var(dim=0) -> unbiased variance along axis 0
    # PyTorch var default is unbiased (N-1), TF reduce_variance is biased (N).
    # We adjust to match PyTorch behavior.
    mean_t5 = tf.reduce_mean(t5, axis=0)
    var_t5_biased = tf.reduce_mean(tf.math.squared_difference(t5, mean_t5), axis=0)
    n_dim0 = tf.cast(tf.shape(t5)[0], tf.float32)
    t6 = var_t5_biased * (n_dim0 / (n_dim0 - 1.0)) # size=(4, 5, 2), dtype=float32, device=cuda
    
    # Conversion: torch.var() -> unbiased scalar variance
    mean_t6 = tf.reduce_mean(t6)
    var_t6_biased = tf.reduce_mean(tf.math.squared_difference(t6, mean_t6))
    n_total = tf.cast(tf.size(t6), tf.float32)
    t7 = var_t6_biased * (n_total / (n_total - 1.0)) # size=(), dtype=float32, device=cuda
    
    t8 = arg4 # size=(), dtype=float32, device=cuda
    
    # Conversion: torch.nn.functional.relu -> tf.nn.relu
    t9 = tf.nn.relu(t8) # size=(), dtype=float32, device=cuda
    
    t10 = t7 + t4 + t9 # size=(), dtype=float32, device=cuda
    
    # Conversion: torch.pow -> tf.pow
    t11 = tf.pow(tf.pow(t4, t7), t10) # size=(), dtype=float32, device=cuda
    output = t11  # output tensor
    return output

# Initialize inputs
# Conversion: torch.rand -> tf.random.uniform
# Conversion: requires_grad=True -> tf.Variable
with tf.device(device):
    arg0 = tf.Variable(tf.random.uniform([5, 4], dtype=tf.bfloat16)) # size=(5, 4), dtype=bfloat16, device=cuda
    arg1 = tf.Variable(tf.random.uniform([5, 1024], dtype=tf.bfloat16)) # size=(5, 1024), dtype=bfloat16, device=cuda
    arg2 = tf.Variable(tf.random.uniform([1024, 4], dtype=tf.bfloat16)) # size=(1024, 4), dtype=bfloat16, device=cuda
    arg3 = tf.Variable(tf.random.uniform([3, 4, 5, 2], dtype=tf.float32)) # size=(3, 4, 5, 2), dtype=float32, device=cuda
    arg4 = tf.Variable(tf.random.uniform([], dtype=tf.float32)) # size=(), dtype=float32, device=cuda

if __name__ == '__main__':
    # Eager execution
    with tf.GradientTape() as tape:
        out_eager = foo(arg0, arg1, arg2, arg3, arg4)
        loss_eager = tf.reduce_sum(out_eager)
    grads_eager = tape.gradient(loss_eager, [arg0, arg1, arg2, arg3, arg4])
    print('Eager Success! ✅')
    
    # Compiled execution
    # Conversion: torch.compile -> tf.function with jit_compile=True
    compiled_foo = tf.function(foo, jit_compile=True)
    
    with tf.GradientTape() as tape:
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4)
        loss_compiled = tf.reduce_sum(out_compiled)
    grads_compiled = tape.gradient(loss_compiled, [arg0, arg1, arg2, arg3, arg4])
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