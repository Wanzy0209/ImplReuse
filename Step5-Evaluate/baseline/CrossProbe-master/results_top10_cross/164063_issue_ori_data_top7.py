```python
import tensorflow as tf
import sys

# PyTorch specific configs are not applicable in TensorFlow
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, arg3, arg4, sentinel):
    t0 = arg0 # size=(36, 7112, 1, 1), dtype=bfloat16
    t1 = tf.reshape(t0, (28, 24, 3, 127)) # size=(28, 24, 3, 127), dtype=bfloat16
    # Note: PyTorch var default is unbiased=True, TF reduce_variance is population variance (unbiased=False)
    t2 = tf.math.reduce_variance(t1, axis=2) # size=(28, 24, 127), dtype=bfloat16
    
    t3 = arg1 # size=(30, 24), dtype=int64
    t4 = arg2 # size=(512, 127), dtype=bfloat16
    
    # Conversion: embedding lookup with clamping
    # torch.nn.functional.embedding(torch.clamp(t3, 0, t4.size(0) - 1).to(torch.long), t4)
    t3_clamped = tf.clip_by_value(t3, 0, tf.shape(t4)[0] - 1)
    t3_cast = tf.cast(t3_clamped, tf.int64)
    t5 = tf.nn.embedding_lookup(t4, t3_cast) # size=(30, 24, 127), dtype=bfloat16
    
    t6 = arg3 # size=(30, 24, 15), dtype=bfloat16
    # Conversion: pad [0, 1] on last dimension -> [[0, 0], [0, 0], [0, 1]]
    t7 = tf.pad(t6, [[0, 0], [0, 0], [0, 1]], mode='CONSTANT', constant_values=0.0) # size=(30, 24, 16), dtype=bfloat16
    
    t8 = arg4 # size=(30, 4, 16, 127), dtype=bfloat16
    t9 = tf.reduce_sum(t8, axis=1) # size=(30, 16, 127), dtype=bfloat16
    
    # Conversion: baddbmm(t5, t7, t9) -> t5 + batch_matmul(t7, t9)
    t10 = t5 + tf.matmul(t7, t9) # size=(30, 24, 127), dtype=bfloat16
    
    t11 = tf.concat([t2, t10], axis=0) # size=(58, 24, 127), dtype=bfloat16
    output = t11 + sentinel  # output tensor with sentinel for gradient flow
    return output

# Input generation
# arg0 = torch.rand([36, 7112, 1, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg0 = tf.random.uniform([36, 7112, 1, 1], dtype=tf.bfloat16)

# arg1 = torch.randint(0, 512, [30, 24], dtype=torch.int64, device='cuda')
arg1 = tf.random.uniform([30, 24], minval=0, maxval=512, dtype=tf.int64)

# arg2 = torch.rand([512, 127], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg2 = tf.random.uniform([512, 127], dtype=tf.bfloat16)

# arg3 = torch.rand([30, 24, 15], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg3 = tf.random.uniform([30, 24, 15], dtype=tf.bfloat16)

# arg4 = torch.rand([30, 4, 16, 127], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg4 = tf.random.uniform([30, 4, 16, 127], dtype=tf.bfloat16)

# sentinel = torch.tensor(0.0, dtype=torch.bfloat16, device='cuda', requires_grad=True)
sentinel = tf.constant(0.0, dtype=tf.bfloat16)

if __name__ == '__main__':
    # Eager execution
    with tf.GradientTape() as tape:
        tape.watch([arg0, arg2, arg3, arg4, sentinel])
        out_eager = foo(arg0, arg1, arg2, arg3, arg4, sentinel)
        loss_eager = tf.reduce_sum(out_eager)
    grads_eager = tape.gradient(loss_eager, [arg0, arg2, arg3, arg4, sentinel])
    print('Eager Success! ✅')
    
    # Compiled execution (tf.function is the TF equivalent of torch.compile)
    compiled_foo = tf.function(foo)
    with tf.GradientTape() as tape:
        tape.watch([arg0, arg2, arg3, arg4, sentinel])
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, sentinel)
        loss_compiled = tf.reduce_sum(out_compiled)
    grads_compiled = tape.gradient(loss_compiled, [arg0, arg2, arg3, arg4, sentinel])
    print('Compile Success! ✅')
```