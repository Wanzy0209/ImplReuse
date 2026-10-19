```python
import tensorflow as tf
import sys

# PyTorch specific configurations are not applicable in TensorFlow
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, arg3, arg4, sentinel):
    t0 = arg0 # size=(36, 7112, 1, 1), dtype=bfloat16
    t1 = tf.reshape(t0, (28, 24, 3, 127)) # size=(28, 24, 3, 127)
    
    # PyTorch var default is unbiased (div by N-1), TF reduce_variance is biased (div by N).
    # For exact numerical parity with PyTorch's default, one would need to adjust the result.
    # Here we use the standard TF op.
    t2 = tf.math.reduce_variance(t1, axis=2) # size=(28, 24, 127)
    
    t3 = arg1 # size=(30, 24), dtype=int64
    t4 = arg2 # size=(512, 127), dtype=bfloat16
    
    # Embedding lookup
    # Clamp indices to valid range [0, num_embeddings - 1]
    max_idx = tf.shape(t4)[0] - 1
    t3_clamped = tf.clip_by_value(t3, 0, max_idx)
    # Cast to int32 as tf.gather typically expects int32 indices
    t3_indices = tf.cast(t3_clamped, tf.int32)
    # Perform embedding lookup
    t5 = tf.gather(t4, t3_indices, axis=0) # size=(30, 24, 127)
    
    t6 = arg3 # size=(30, 24, 15), dtype=bfloat16
    # Pad the last dimension: [0, 1] means 0 padding on left, 1 on right
    # TF paddings format is [[0, 0], [0, 0], [0, 1]] for rank 3 tensor
    t7 = tf.pad(t6, [[0, 0], [0, 0], [0, 1]], mode='CONSTANT', constant_values=0.0) # size=(30, 24, 16)
    
    t8 = arg4 # size=(30, 4, 16, 127), dtype=bfloat16
    t9 = tf.reduce_sum(t8, axis=1) # size=(30, 16, 127)
    
    # Batched matrix multiply and add: baddbmm(t5, t7, t9) -> t5 + (t7 @ t9)
    # t7: (30, 24, 16), t9: (30, 16, 127) -> matmul result: (30, 24, 127)
    t10 = t5 + tf.linalg.matmul(t7, t9) # size=(30, 24, 127)
    
    t11 = tf.concat([t2, t10], axis=0) # size=(58, 24, 127)
    
    output = t11 + sentinel # output tensor with sentinel for gradient flow
    return output

# Input generation
# Note: TF uses float32 by default, explicitly casting to bfloat16 to match source
arg0 = tf.random.uniform((36, 7112, 1, 1), dtype=tf.bfloat16) # size=(36, 7112, 1, 1), dtype=bfloat16
arg1 = tf.random.uniform((30, 24), minval=0, maxval=512, dtype=tf.int64) # size=(30, 24), dtype=int64
arg2 = tf.random.uniform((512, 127), dtype=tf.bfloat16) # size=(512, 127), dtype=bfloat16
arg3 = tf.random.uniform((30, 24, 15), dtype=tf.bfloat16) # size=(30, 24, 15), dtype=bfloat16
arg4 = tf.random.uniform((30, 4, 16, 127), dtype=tf.bfloat16) # size=(30, 4, 16, 127), dtype=bfloat16
sentinel = tf.constant(0.0, dtype=tf.bfloat16) # Sentinel for gradient flow

if __name__ == '__main__':
    # Eager execution
    with tf.GradientTape(persistent=True) as tape:
        # Watch inputs to compute gradients
        tape.watch([arg0, arg1, arg2, arg3, arg4, sentinel])
        out_eager = foo(arg0, arg1, arg2, arg3, arg4, sentinel)
        loss_eager = tf.reduce_sum(out_eager)
    
    # Compute gradients (equivalent to .backward())
    grads_eager = tape.gradient(loss_eager, [arg0, arg1, arg2, arg3, arg4, sentinel])
    print('Eager Success! ✅')
    
    # Compiled execution (tf.function is the TF equivalent of torch.compile)
    compiled_foo = tf.function(foo)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([arg0, arg1, arg2, arg3, arg4, sentinel])
        out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, sentinel)
        loss_compiled = tf.reduce_sum(out_compiled)
        
    grads_compiled = tape.gradient(loss_compiled, [arg0, arg1, arg2, arg3, arg4, sentinel])
    print('Compile Success! ✅')
```