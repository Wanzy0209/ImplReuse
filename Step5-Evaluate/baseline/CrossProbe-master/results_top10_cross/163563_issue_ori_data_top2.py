```python
import tensorflow as tf
import sys

# PyTorch specific configs removed (no direct TF equivalent)
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(5699097, 6, 1), stride=(6, 1, 1), dtype=bfloat16, device=cuda
    t1 = tf.sigmoid(t0) # size=(5699097, 6, 1), stride=(6, 1, 1), dtype=bfloat16, device=cuda
    t2 = arg1 # size=(5699097, 6, 256), stride=(1536, 256, 1), dtype=bfloat16, device=cuda
    t3 = tf.sigmoid(t2) # size=(5699097, 6, 256), stride=(1536, 256, 1), dtype=bfloat16, device=cuda
    t4 = arg2 # size=(5699097, 256, 1), stride=(256, 1, 1), dtype=bfloat16, device=cuda
    t5 = tf.exp(t4) # size=(5699097, 256, 1), stride=(256, 1, 1), dtype=bfloat16, device=cuda
    
    # Conversion: torch.baddbmm(t1, t3, t5) -> t1 + batch_matmul(t3, t5)
    # t3: (B, 6, 256), t5: (B, 256, 1) -> matmul: (B, 6, 1)
    # t1: (B, 6, 1)
    t6 = t1 + tf.linalg.matmul(t3, t5) # size=(5699097, 6, 1), stride=(6, 1, 1), dtype=bfloat16, device=cuda
    
    t7 = tf.reshape(t6, (193, 386, 459)) # size=(193, 386, 459), stride=(177174, 459, 1), dtype=bfloat16, device=cuda
    output = t7  # output tensor
    return output

# Conversion: torch.rand -> tf.random.uniform
# Note: requires_grad is handled via GradientTape in TF, not tensor creation
arg0 = tf.random.uniform([5699097, 6, 1], dtype=tf.bfloat16) # size=(5699097, 6, 1), stride=(6, 1, 1), dtype=bfloat16, device=cuda
arg1 = tf.random.uniform([5699097, 6, 256], dtype=tf.bfloat16) # size=(5699097, 6, 256), stride=(1536, 256, 1), dtype=bfloat16, device=cuda
arg2 = tf.random.uniform([5699097, 256, 1], dtype=tf.bfloat16) # size=(5699097, 256, 1), stride=(256, 1, 1), dtype=bfloat16, device=cuda

if __name__ == '__main__':
    # Eager execution
    with tf.GradientTape() as tape:
        tape.watch([arg0, arg1, arg2])
        out_eager = foo(arg0, arg1, arg2)
        # Conversion: .sum().backward() -> reduce_sum + tape.gradient
        loss = tf.reduce_sum(out_eager)
    grads = tape.gradient(loss, [arg0, arg1, arg2])
    print('Eager Success! ✅')
    
    # Conversion: torch.compile -> tf.function
    # fullgraph=True maps to tf.function creating a single graph
    compiled_foo = tf.function(foo, jit_compile=True)
    
    with tf.GradientTape() as tape:
        tape.watch([arg0, arg1, arg2])
        out_compiled = compiled_foo(arg0, arg1, arg2)
        loss = tf.reduce_sum(out_compiled)
    grads = tape.gradient(loss, [arg0, arg1, arg2])
    print('Compile Success! ✅')
```