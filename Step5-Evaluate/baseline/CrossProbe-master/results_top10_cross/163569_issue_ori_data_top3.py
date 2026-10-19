```python
import tensorflow as tf
import sys

# PyTorch specific configs (capture_scalar_outputs, etc.) are omitted as they 
# do not have direct equivalents in TensorFlow's eager or graph execution modes.

def foo(arg0, arg1, arg2):
    # t0 = arg0 # size=(2, 261, 17, 358), stride=(1588446, 6086, 358, 1), dtype=bfloat16, device=cuda
    t0 = arg0
    
    # t1 = t0.max(dim=0).values # size=(261, 17, 358), stride=(358, 93438, 1), dtype=bfloat16, device=cuda
    # Conversion: torch.max(dim=0) -> tf.reduce_max(axis=0)
    t1 = tf.reduce_max(t0, axis=0)
    
    # t2 = t1.transpose(1, 0) # size=(17, 261, 358), stride=(93438, 358, 1), dtype=bfloat16, device=cuda
    # Conversion: torch.transpose(1, 0) on 3D tensor -> tf.transpose with perm=[1, 0, 2]
    t2 = tf.transpose(t1, perm=[1, 0, 2])
    
    # t3 = arg1 # size=(17, 64, 358), stride=(1088, 64, 1), dtype=float32, device=cuda
    t3 = arg1
    
    # t4 = torch.exp(t3) # size=(17, 64, 358), stride=(1088, 64, 1), dtype=float32, device=cuda
    t4 = tf.exp(t3)
    
    # t5 = arg2 # size=(261, 1, 64), stride=(16704, 1, 64), dtype=float32, device=cuda
    t5 = arg2
    
    # t6 = t5.transpose(2, 1) # size=(261, 64, 1), stride=(16704, 64, 1), dtype=float32, device=cuda
    # Conversion: torch.transpose(2, 1) on 3D tensor -> tf.transpose with perm=[0, 2, 1]
    t6 = tf.transpose(t5, perm=[0, 2, 1])
    
    # t7 = torch.nn.functional.conv1d(t4, t6, stride=1, padding=0) # size=(17, 261, 358), stride=(93438, 358, 1), dtype=float32, device=cuda
    # Conversion: PyTorch conv1d uses (N, C_in, L) and weights (C_out, C_in, K).
    # TensorFlow conv1d uses (N, L, C_in) and filters (K, C_in, C_out).
    # t4 shape: (17, 64, 358) -> Transpose to (17, 358, 64)
    # t6 shape: (261, 64, 1) -> Transpose to (1, 64, 261)
    # Output shape: (17, 358, 261) -> Transpose back to (17, 261, 358) to match PyTorch
    t4_transposed = tf.transpose(t4, perm=[0, 2, 1])
    t6_transposed = tf.transpose(t6, perm=[2, 1, 0])
    t7_intermediate = tf.nn.conv1d(t4_transposed, t6_transposed, stride=1, padding='VALID')
    t7 = tf.transpose(t7_intermediate, perm=[0, 2, 1])
    
    # t8 = t7.clone(); t8.zero_() # size=(17, 261, 358), stride=(93438, 358, 1), dtype=float32, device=cuda
    # Conversion: clone() and zero_() -> tf.zeros_like()
    t8 = tf.zeros_like(t7)
    
    # t9 = t2 * t7 * t8 # size=(17, 261, 358), stride=(93438, 358, 1), dtype=float32, device=cuda
    t9 = t2 * t7 * t8
    
    output = t9  # output tensor
    return output

# Conversion: torch.rand with device='cuda' -> tf.random.uniform inside tf.device context
with tf.device('/GPU:0'):
    arg0 = tf.random.uniform([2, 261, 17, 358], dtype=tf.bfloat16) # size=(2, 261, 17, 358), dtype=bfloat16
    arg1 = tf.random.uniform([17, 64, 358], dtype=tf.float32) # size=(17, 64, 358), dtype=float32
    arg2 = tf.random.uniform([261, 1, 64], dtype=tf.float32) # size=(261, 1, 64), dtype=float32

if __name__ == '__main__':
    # Eager execution
    with tf.GradientTape() as tape:
        tape.watch([arg0, arg1, arg2])
        out_eager = foo(arg0, arg1, arg2)
        loss = tf.reduce_sum(out_eager)
    grads = tape.gradient(loss, [arg0, arg1, arg2])
    print('Eager Success! ✅')
    
    # Compiled execution (tf.function is the TF equivalent of torch.compile for graph optimization)
    compiled_foo = tf.function(foo)
    with tf.GradientTape() as tape:
        tape.watch([arg0, arg1, arg2])
        out_compiled = compiled_foo(arg0, arg1, arg2)
        loss = tf.reduce_sum(out_compiled)
    grads = tape.gradient(loss, [arg0, arg1, arg2])
    print('Compile Success! ✅')
```