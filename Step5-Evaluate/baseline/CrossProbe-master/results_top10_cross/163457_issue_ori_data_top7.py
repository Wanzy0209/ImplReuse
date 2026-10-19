```python
import tensorflow as tf
import sys

# PyTorch specific configurations removed as they do not apply to TensorFlow
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    # t0 = arg0 # size=(3, 395, 202, 357), stride=(239370, 79790, 202, 1), dtype=bfloat16, device=cuda
    # t1 = arg1 # size=(1,), stride=(1,), dtype=int64, device=cuda
    # t2 = arg2 # size=(395,), stride=(1,), dtype=bfloat16, device=cuda
    
    # Conversion: torch.nn.functional.group_norm with groups=1
    # This normalizes over the channel and spatial dimensions (C, H, W) for each batch item.
    # We implement this manually using tf.nn.moments to match the functional API behavior
    # and allow passing explicit weight/bias tensors.
    # Input shape: (N, C, H, W) = (3, 395, 202, 357)
    mean, variance = tf.nn.moments(arg0, axes=[1, 2, 3], keepdims=True)
    epsilon = 1e-5
    t3 = (arg0 - mean) * tf.math.rsqrt(variance + epsilon)
    
    # Apply weight and bias (t2 is used for both)
    # t2 shape (395,) broadcasts to (3, 395, 202, 357)
    t3 = t3 * t2 + t2 # size=(3, 395, 202, 357), dtype=bfloat16, device=cuda
    
    # Conversion: t3.max(dim=0).values
    # Reduces over the batch dimension (axis 0)
    t4 = tf.reduce_max(t3, axis=0) # size=(395, 202, 357), dtype=bfloat16, device=cuda
    
    # Conversion: torch.sigmoid(t4)
    t5 = tf.nn.sigmoid(t4) # size=(395, 202, 357), dtype=bfloat16, device=cuda
    
    output = t5  # output tensor
    return output

# Use GPU if available, matching the 'cuda' device in the source
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'
with tf.device(device):
    # Conversion: torch.rand([3, 395, 202, 357], dtype=torch.bfloat16, device='cuda', requires_grad=True)
    arg0 = tf.Variable(tf.random.uniform([3, 395, 202, 357], dtype=tf.bfloat16))
    
    # Conversion: torch.randint(0, 1000, [1], dtype=torch.int64, device='cuda')
    arg1 = tf.Variable(tf.random.uniform([1], minval=0, maxval=1000, dtype=tf.int64))
    
    # Conversion: torch.rand([395], dtype=torch.bfloat16, device='cuda', requires_grad=True)
    arg2 = tf.Variable(tf.random.uniform([395], dtype=tf.bfloat16))

if __name__ == '__main__':
    # Eager execution
    with tf.GradientTape() as tape:
        out_eager = foo(arg0, arg1, arg2)
        # PyTorch: out_eager.sum().backward()
        # We compute the sum to match the logic, though gradients aren't explicitly compared in the print statements
        loss_eager = tf.reduce_sum(out_eager)
    
    # Compute gradients to ensure the backward pass is executed
    grads_eager = tape.gradient(loss_eager, [arg0, arg1, arg2])
    print('Eager Success! ✅')
    
    # Conversion: torch.compile(foo, fullgraph=True, dynamic=True)
    # In TensorFlow, tf.function compiles the function into a graph.
    # fullgraph=True is roughly analogous to tf.function's default behavior of tracing.
    compiled_foo = tf.function(foo)
    
    with tf.GradientTape() as tape:
        out_compiled = compiled_foo(arg0, arg1, arg2)
        loss_compiled = tf.reduce_sum(out_compiled)
        
    grads_compiled = tape.gradient(loss_compiled, [arg0, arg1, arg2])
    print('Compile Success! ✅')
    
    # Compare outputs (forward)
    # Conversion: out_eager.sum()
    out_eager_sum = tf.reduce_sum(out_eager)
    out_compiled_sum = tf.reduce_sum(out_compiled)
    
    # Conversion: (out_eager_sum - out_compiled_sum).abs().item()
    diff = tf.abs(out_eager_sum - out_compiled_sum)
    # Conversion: diff / (out_eager_sum.abs().item() + 1e-12) * 100
    rel_diff = diff / (tf.abs(out_eager_sum) + 1e-12) * 100
    
    # Convert tensors to numpy scalars for printing
    diff_val = diff.numpy()
    rel_diff_val = rel_diff.numpy()
    out_eager_sum_val = out_eager_sum.numpy()
    out_compiled_sum_val = out_compiled_sum.numpy()

    print(f'Relative diff (sum): {rel_diff_val:.6f}%')
    
    if rel_diff_val > 5:
        print(f'❌ Forward output sums differ significantly (relative)!')
        print('out_eager_sum:', out_eager_sum_val)
        print('out_compiled_sum:', out_compiled_sum_val)
        print('Absolute diff:', diff_val)
        print('Relative diff (%):', rel_diff_val)
        sys.exit(1)
```