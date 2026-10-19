```python
import tensorflow as tf
import sys

# PyTorch specific configurations are omitted as they do not apply to TensorFlow
# torch._dynamo.config.capture_scalar_outputs = True
# torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(2, 261, 17, 358), dtype=bfloat16
    # Conversion: torch.max(dim=0).values -> tf.reduce_max(axis=0)
    t1 = tf.reduce_max(t0, axis=0) # size=(261, 17, 358), dtype=bfloat16
    # Conversion: torch.transpose(1, 0) -> tf.transpose(perm=[1, 0, 2])
    t2 = tf.transpose(t1, perm=[1, 0, 2]) # size=(17, 261, 358), dtype=bfloat16
    
    t3 = arg1 # size=(17, 64, 358), dtype=float32
    t4 = tf.exp(t3) # size=(17, 64, 358), dtype=float32
    
    t5 = arg2 # size=(261, 1, 64), dtype=float32
    # Conversion: torch.transpose(2, 1) -> tf.transpose(perm=[0, 2, 1])
    t6 = tf.transpose(t5, perm=[0, 2, 1]) # size=(261, 64, 1), dtype=float32
    
    # Conversion: torch.nn.functional.conv1d
    # PyTorch conv1d input: (Batch, Channels, Length), Weight: (OutChannels, InChannels, KernelSize)
    # TensorFlow conv1d input: (Batch, Length, Channels), Filter: (KernelSize, InChannels, OutChannels)
    
    # Transpose t4 from (17, 64, 358) to (17, 358, 64)
    t4_transposed = tf.transpose(t4, perm=[0, 2, 1])
    
    # Transpose t6 from (261, 64, 1) to (1, 64, 261)
    # PyTorch weight shape (Out, In, K) -> TF filter shape (K, In, Out)
    t6_transposed = tf.transpose(t6, perm=[2, 1, 0])
    
    # Perform convolution
    conv_out = tf.nn.conv1d(t4_transposed, t6_transposed, stride=1, padding='VALID')
    
    # Transpose output back to (17, 261, 358) to match PyTorch shape
    t7 = tf.transpose(conv_out, perm=[0, 2, 1]) # size=(17, 261, 358), dtype=float32
    
    # Conversion: t7.clone(); t8.zero_() -> tf.zeros_like(t7)
    t8 = tf.zeros_like(t7) # size=(17, 261, 358), dtype=float32
    
    # Cast t2 to float32 to match t7 and t8 for multiplication
    t2 = tf.cast(t2, tf.float32)
    
    t9 = t2 * t7 * t8 # size=(17, 261, 358), dtype=float32
    output = t9  # output tensor
    return output

# Conversion: torch.rand -> tf.random.uniform
# Note: Device placement is handled automatically by TensorFlow
arg0 = tf.random.uniform([2, 261, 17, 358], dtype=tf.bfloat16) # size=(2, 261, 17, 358), dtype=bfloat16
arg1 = tf.random.uniform([17, 64, 358], dtype=tf.float32) # size=(17, 64, 358), dtype=float32
arg2 = tf.random.uniform([261, 1, 64], dtype=tf.float32) # size=(261, 1, 64), dtype=float32

if __name__ == '__main__':
    # Eager Execution with GradientTape
    with tf.GradientTape() as tape:
        tape.watch([arg0, arg1, arg2])
        out_eager = foo(arg0, arg1, arg2)
        loss = tf.reduce_sum(out_eager)
    grads_eager = tape.gradient(loss, [arg0, arg1, arg2])
    print('Eager Success! ✅')
    
    # Compiled Execution using tf.function
    compiled_foo = tf.function(foo)
    with tf.GradientTape() as tape:
        tape.watch([arg0, arg1, arg2])
        out_compiled = compiled_foo(arg0, arg1, arg2)
        loss = tf.reduce_sum(out_compiled)
    grads_compiled = tape.gradient(loss, [arg0, arg1, arg2])
    print('Compile Success! ✅')
```