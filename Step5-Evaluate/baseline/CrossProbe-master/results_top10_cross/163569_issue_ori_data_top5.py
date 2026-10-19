```python
import tensorflow as tf
import sys

# PyTorch specific configs (torch._dynamo, torch._inductor) are omitted as they are framework-specific.
# TensorFlow uses tf.function for graph compilation and optimization.

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(2, 261, 17, 358), stride=(1588446, 6086, 358, 1), dtype=bfloat16, device=cuda
    t1 = tf.reduce_max(t0, axis=0) # size=(261, 17, 358), stride=(358, 93438, 1), dtype=bfloat16, device=cuda
    t2 = tf.transpose(t1, perm=[1, 0, 2]) # size=(17, 261, 358), stride=(93438, 358, 1), dtype=bfloat16, device=cuda
    t3 = arg1 # size=(17, 64, 358), stride=(1088, 64, 1), dtype=float32, device=cuda
    t4 = tf.exp(t3) # size=(17, 64, 358), stride=(1088, 64, 1), dtype=float32, device=cuda
    t5 = arg2 # size=(261, 1, 64), stride=(16704, 1, 64), dtype=float32, device=cuda
    t6 = tf.transpose(t5, perm=[0, 2, 1]) # size=(261, 64, 1), stride=(16704, 64, 1), dtype=float32, device=cuda
    
    # Conversion for conv1d:
    # PyTorch conv1d input format: (Batch, Channels, Length)
    # PyTorch conv1d weight format: (Out_Channels, In_Channels, Kernel_Size)
    # TensorFlow conv1d input format: (Batch, Length, Channels)
    # TensorFlow conv1d filter format: (Kernel_Size, In_Channels, Out_Channels)
    
    # t4: (17, 64, 358) -> Transpose to (17, 358, 64)
    t4_transposed = tf.transpose(t4, perm=[0, 2, 1])
    # t6: (261, 64, 1) -> Transpose to (1, 64, 261)
    t6_transposed = tf.transpose(t6, perm=[2, 1, 0])
    
    # stride=1, padding=0 corresponds to padding='VALID' in TensorFlow
    conv_out = tf.nn.conv1d(t4_transposed, t6_transposed, stride=1, padding='VALID')
    
    # Output of TF conv1d is (17, 358, 261). Transpose back to (17, 261, 358) to match PyTorch output t7
    t7 = tf.transpose(conv_out, perm=[0, 2, 1]) # size=(17, 261, 358), stride=(93438, 358, 1), dtype=float32, device=cuda
    
    t8 = tf.identity(t7) # clone
    t8 = tf.zeros_like(t8) # zero_
    t9 = t2 * t7 * t8 # size=(17, 261, 358), stride=(93438, 358, 1), dtype=float32, device=cuda
    output = t9  # output tensor
    return output

# Use tf.device to ensure execution on GPU, mirroring device='cuda'
with tf.device('/GPU:0'):
    arg0 = tf.random.uniform([2, 261, 17, 358], dtype=tf.bfloat16) # size=(2, 261, 17, 358), stride=(1588446, 6086, 358, 1), dtype=bfloat16, device=cuda
    arg1 = tf.random.uniform([17, 64, 358], dtype=tf.float32) # size=(17, 64, 358), stride=(1088, 64, 1), dtype=float32, device=cuda
    arg2 = tf.random.uniform([261, 1, 64], dtype=tf.float32) # size=(261, 1, 64), stride=(16704, 1, 64), dtype=float32, device=cuda

if __name__ == '__main__':
    # Eager execution
    with tf.GradientTape() as tape:
        # Watch inputs to calculate gradients (equivalent to requires_grad=True)
        tape.watch([arg0, arg1, arg2])
        out_eager = foo(arg0, arg1, arg2)
        loss = tf.reduce_sum(out_eager)
    grads = tape.gradient(loss, [arg0, arg1, arg2])
    print('Eager Success! ✅')
    
    # Compiled execution (tf.function is the TF equivalent of torch.compile)
    compiled_foo = tf.function(foo)
    
    with tf.GradientTape() as tape:
        tape.watch([arg0, arg1, arg2])
        out_compiled = compiled_foo(arg0, arg1, arg2)
        loss = tf.reduce_sum(out_compiled)
    grads = tape.gradient(loss, [arg0, arg1, arg2])
    print('Compile Success! ✅')
```