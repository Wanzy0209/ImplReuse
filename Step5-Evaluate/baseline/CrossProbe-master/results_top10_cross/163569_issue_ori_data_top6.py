```python
import tensorflow as tf
import sys

# Note: PyTorch dynamo/inductor configs are specific to PyTorch compilation.
# TensorFlow uses XLA or graph compilation via @tf.function.
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(2, 261, 17, 358), stride=(1588446, 6086, 358, 1), dtype=bfloat16, device=cuda
    t1 = tf.reduce_max(t0, axis=0) # size=(261, 17, 358), stride=(358, 93438, 1), dtype=bfloat16, device=cuda
    t2 = tf.transpose(t1, perm=[1, 0, 2]) # size=(17, 261, 358), stride=(93438, 358, 1), dtype=bfloat16, device=cuda
    t3 = arg1 # size=(17, 64, 358), stride=(1088, 64, 1), dtype=float32, device=cuda
    t4 = tf.exp(t3) # size=(17, 64, 358), stride=(1088, 64, 1), dtype=float32, device=cuda
    t5 = arg2 # size=(261, 1, 64), stride=(16704, 1, 64), dtype=float32, device=cuda
    t6 = tf.transpose(t5, perm=[0, 2, 1]) # size=(261, 64, 1), stride=(16704, 64, 1), dtype=float32, device=cuda
    
    # Conversion: PyTorch conv1d uses (N, C, L), TF uses (N, L, C) by default.
    # Transposing inputs/weights to match TF expectations.
    t4_transposed = tf.transpose(t4, perm=[0, 2, 1]) # (17, 358, 64)
    t6_filter = tf.transpose(t6, perm=[2, 1, 0])     # (1, 64, 261) -> (K, C_in, C_out)
    
    # t7 = torch.nn.functional.conv1d(t4, t6, stride=1, padding=0)
    t7_transposed = tf.nn.conv1d(t4_transposed, t6_filter, stride=1, padding='VALID', data_format='NWC')
    t7 = tf.transpose(t7_transposed, perm=[0, 2, 1]) # size=(17, 261, 358), stride=(93438, 358, 1), dtype=float32, device=cuda
    
    # t8 = t7.clone(); t8.zero_()
    # TF tensors are immutable, so we create a new zero tensor.
    t8 = tf.zeros_like(t7) # size=(17, 261, 358), stride=(93438, 358, 1), dtype=float32, device=cuda
    
    t9 = t2 * t7 * t8 # size=(17, 261, 358), stride=(93438, 358, 1), dtype=float32, device=cuda
    output = t9  # output tensor
    return output

# arg0 = torch.rand([2, 261, 17, 358], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg0 = tf.random.uniform([2, 261, 17, 358], dtype=tf.bfloat16) # size=(2, 261, 17, 358), stride=(1588446, 6086, 358, 1), dtype=bfloat16, device=cuda
# arg1 = torch.rand([17, 64, 358], dtype=torch.float32, device='cuda', requires_grad=True)
arg1 = tf.random.uniform([17, 64, 358], dtype=tf.float32) # size=(17, 64, 358), stride=(1088, 64, 1), dtype=float32, device=cuda
# arg2 = torch.rand([261, 1, 64], dtype=torch.float32, device='cuda', requires_grad=True)
arg2 = tf.random.uniform([261, 1, 64], dtype=tf.float32) # size=(261, 1, 64), stride=(16704, 1, 64), dtype=float32, device=cuda

if __name__ == '__main__':
    # Eager execution
    with tf.GradientTape() as tape:
        tape.watch([arg0, arg1, arg2])
        out_eager = foo(arg0, arg1, arg2)
        loss = tf.reduce_sum(out_eager)
    grads = tape.gradient(loss, [arg0, arg1, arg2])
    print('Eager Success! ✅')
    
    # Compiled execution (using tf.function)
    compiled_foo = tf.function(foo)
    with tf.GradientTape() as tape:
        tape.watch([arg0, arg1, arg2])
        out_compiled = compiled_foo(arg0, arg1, arg2)
        loss = tf.reduce_sum(out_compiled)
    grads = tape.gradient(loss, [arg0, arg1, arg2])
    print('Compile Success! ✅')
```