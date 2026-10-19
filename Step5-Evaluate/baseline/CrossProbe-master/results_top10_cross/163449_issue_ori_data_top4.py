```python
import tensorflow as tf
import sys

# PyTorch specific compilation configs are omitted in TensorFlow.
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, arg3, arg4):
    t0 = arg0 # size=(5, 4), dtype=bfloat16, device=cuda
    t1 = arg1 # size=(5, 1024), dtype=bfloat16, device=cuda
    t2 = arg2 # size=(1024, 4), dtype=bfloat16, device=cuda
    
    # torch.addmm(t0, t1, t2) -> t0 + t1 @ t2
    t3 = t0 + tf.linalg.matmul(t1, t2) # size=(5, 4), dtype=bfloat16, device=cuda
    
    # t3.norm() -> L2 norm (Frobenius norm for matrices)
    t4 = tf.norm(t3) # size=(), dtype=bfloat16, device=cuda
    
    t5 = arg3 # size=(3, 4, 5, 2), dtype=float32, device=cuda
    
    # t5.var(dim=0) -> Variance along axis 0
    # Note: PyTorch var default is unbiased (N-1), TF reduce_variance is biased (N).
    # This may cause slight numerical differences compared to PyTorch.
    t6 = tf.math.reduce_variance(t5, axis=0) # size=(4, 5, 2), dtype=float32, device=cuda
    
    # t6.var() -> Scalar variance
    t7 = tf.math.reduce_variance(t6) # size=(), dtype=float32, device=cuda
    
    t8 = arg4 # size=(), dtype=float32, device=cuda
    
    # torch.nn.functional.relu(t8)
    t9 = tf.nn.relu(t8) # size=(), dtype=float32, device=cuda
    
    t10 = t7 + t4 + t9 # size=(), dtype=float32, device=cuda
    
    # torch.pow(torch.pow(t4, t7), t10)
    t11 = tf.math.pow(tf.math.pow(t4, t7), t10) # size=(), dtype=float32, device=cuda
    
    output = t11  # output tensor
    return output

# Initialize inputs as Variables to mimic requires_grad=True
# Using tf.device('/GPU:0') to match device='cuda'
with tf.device('/GPU:0'):
    arg0 = tf.Variable(tf.random.uniform([5, 4], dtype=tf.bfloat16)) # size=(5, 4), dtype=bfloat16, device=cuda
    arg1 = tf.Variable(tf.random.uniform([5, 1024], dtype=tf.bfloat16)) # size=(5, 1024), dtype=bfloat16, device=cuda
    arg2 = tf.Variable(tf.random.uniform([1024, 4], dtype=tf.bfloat16)) # size=(1024, 4), dtype=bfloat16, device=cuda
    arg3 = tf.Variable(tf.random.uniform([3, 4, 5, 2], dtype=tf.float32)) # size=(3, 4, 5, 2), dtype=float32, device=cuda
    arg4 = tf.Variable(tf.random.uniform([], dtype=tf.float32)) # size=(), dtype=float32, device=cuda

if __name__ == '__main__':
    # Eager execution
    with tf.GradientTape() as tape_eager:
        out_eager = foo(arg0, arg1, arg2, arg3, arg4)
    
    # Mimic .backward() by computing gradients of the sum
    grads_eager = tape_eager.gradient(out_eager, [arg0, arg1, arg2, arg3, arg4])
    print('Eager Success! ✅')
    
    # Compiled execution using tf.function
    compiled_foo = tf.function(foo)
    with tf.GradientTape() as tape_compiled:
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4)
        
    grads_compiled = tape_compiled.gradient(out_compiled, [arg0, arg1, arg2, arg3, arg4])
    print('Compile Success! ✅')
    
    # Compare outputs (forward)
    out_eager_sum = tf.reduce_sum(out_eager)
    out_compiled_sum = tf.reduce_sum(out_compiled)
    
    diff = tf.abs(out_eager_sum - out_compiled_sum)
    rel_diff = diff / (tf.abs(out_eager_sum) + 1e-12) * 100
    
    print(f'Relative diff (sum): {rel_diff:.6f}%')
    
    if rel_diff > 5:
        print(f'❌ Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', out_eager_sum.numpy())
        print('out_compiled_sum:', out_compiled_sum.numpy())
        print('Absolute diff:', diff.numpy())
        print('Relative diff (%):', rel_diff.numpy())
        sys.exit(1)
```