```python
import tensorflow as tf
import sys
import numpy as np

# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._inductor.config.emulate_precision_casts = True
# Note: PyTorch compilation configs are specific to torch.compile/torchinductor.
# In TensorFlow, tf.function handles graph compilation and optimization.

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(3, 395, 202, 357), stride=(239370, 79790, 202, 1), dtype=bfloat16, device=cuda
    t1 = arg1 # size=(1,), stride=(1,), dtype=int64, device=cuda
    t2 = arg2 # size=(395,), stride=(1,), dtype=bfloat16, device=cuda
    
    # t3 = torch.nn.functional.group_norm(t0, 1, weight=t2, bias=t2)
    # Conversion: Manual implementation of GroupNorm with num_groups=1.
    # Normalizes over C*H*W, applies affine params (weight/bias) to C.
    N = tf.shape(t0)[0]
    C = tf.shape(t0)[1]
    # Reshape to (N, groups, -1) -> (N, 1, C*H*W)
    t0_reshaped = tf.reshape(t0, [N, 1, -1])
    # Calculate mean and variance over the last dimension
    mean, var = tf.nn.moments(t0_reshaped, axes=[2], keepdims=True)
    epsilon = 1e-5 # PyTorch default for group_norm
    t3_reshaped = (t0_reshaped - mean) / tf.sqrt(var + epsilon)
    # Reshape back to (N, C, H, W)
    t3 = tf.reshape(t3_reshaped, tf.shape(t0))
    # Apply weight and bias (broadcasting over N, H, W)
    t3 = t3 * t2 + t2

    t4 = tf.reduce_max(t3, axis=0) # t3.max(dim=0).values
    t5 = tf.sigmoid(t4) # torch.sigmoid(t4)
    output = t5  # output tensor
    return output

# arg0 = torch.rand([3, 395, 202, 357], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg0 = tf.random.uniform((3, 395, 202, 357), dtype=tf.bfloat16)

# arg1 = torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda')
arg1 = tf.random.uniform((1,), minval=0, maxval=1000, dtype=tf.int64)

# arg2 = torch.rand([395], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg2 = tf.random.uniform((395,), dtype=tf.bfloat16)

if __name__ == '__main__':
    # Eager execution
    with tf.GradientTape() as tape:
        tape.watch(arg0)
        tape.watch(arg2)
        out_eager = foo(arg0, arg1, arg2)
        out_eager_sum = tf.reduce_sum(out_eager)
    
    # Compute gradients to ensure graph validity (equivalent to .backward())
    grads = tape.gradient(out_eager_sum, [arg0, arg2])
    print('Eager Success! ✅')

    # Compiled execution
    # tf.function is the TensorFlow equivalent of torch.compile
    compiled_foo = tf.function(foo)
    
    with tf.GradientTape() as tape:
        tape.watch(arg0)
        tape.watch(arg2)
        out_compiled = compiled_foo(arg0, arg1, arg2)
        out_compiled_sum = tf.reduce_sum(out_compiled)
        
    grads = tape.gradient(out_compiled_sum, [arg0, arg2])
    print('Compile Success! ✅')

    # Compare outputs (forward)
    out_eager_sum_val = out_eager_sum.numpy()
    out_compiled_sum_val = out_compiled_sum.numpy()
    
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