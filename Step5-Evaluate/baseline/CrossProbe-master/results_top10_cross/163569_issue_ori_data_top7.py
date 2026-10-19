```python
import tensorflow as tf
# import sys # Not strictly needed for TF logic but kept for structure

# PyTorch specific configs are omitted as they don't translate directly
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(2, 261, 17, 358), stride=(1588446, 6086, 358, 1), dtype=bfloat16, device=cuda
    # Conversion: torch.max(dim=0).values -> tf.reduce_max(axis=0)
    t1 = tf.reduce_max(t0, axis=0) # size=(261, 17, 358), stride=(358, 93438, 1), dtype=bfloat16, device=cuda
    # Conversion: torch.transpose(1, 0) -> tf.transpose(perm=[1, 0, 2])
    t2 = tf.transpose(t1, perm=[1, 0, 2]) # size=(17, 261, 358), stride=(93438, 358, 1), dtype=bfloat16, device=cuda
    t3 = arg1 # size=(17, 64, 358), stride=(1088, 64, 1), dtype=float32, device=cuda
    t4 = tf.exp(t3) # size=(17, 64, 358), stride=(1088, 64, 1), dtype=float32, device=cuda
    t5 = arg2 # size=(261, 1, 64), stride=(16704, 1, 64), dtype=float32, device=cuda
    # Conversion: torch.transpose(2, 1) -> tf.transpose(perm=[0, 2, 1])
    t6 = tf.transpose(t5, perm=[0, 2, 1]) # size=(261, 64, 1), stride=(16704, 64, 1), dtype=float32, device=cuda
    
    # Conversion: torch.nn.functional.conv1d
    # PyTorch input: (Batch, Channels, Length) -> (17, 64, 358)
    # PyTorch weight: (OutChannels, InChannels, KernelSize) -> (261, 64, 1)
    # TF conv1d with data_format='NCW' expects input (Batch, Channels, Length)
    # TF conv1d expects filter (KernelSize, InChannels, OutChannels) -> (1, 64, 261)
    # We transpose t6 from (261, 64, 1) to (1, 64, 261)
    t7 = tf.nn.conv1d(input=t4, filters=tf.transpose(t6, [2, 1, 0]), stride=1, padding='VALID', data_format='NCW') # size=(17, 261, 358), stride=(93438, 358, 1), dtype=float32, device=cuda
    
    # Conversion: t7.clone(); t8.zero_()
    # TF tensors are immutable, so we create a new zero tensor of the same shape
    t8 = tf.zeros_like(t7) # size=(17, 261, 358), stride=(93438, 358, 1), dtype=float32, device=cuda
    
    t9 = t2 * t7 * t8 # size=(17, 261, 358), stride=(93438, 358, 1), dtype=float32, device=cuda
    output = t9  # output tensor
    return output

# Conversion: torch.rand -> tf.random.uniform
# Conversion: requires_grad=True -> tf.Variable
arg0 = tf.Variable(tf.random.uniform([2, 261, 17, 358], dtype=tf.bfloat16)) # size=(2, 261, 17, 358), stride=(1588446, 6086, 358, 1), dtype=bfloat16, device=cuda
arg1 = tf.Variable(tf.random.uniform([17, 64, 358], dtype=tf.float32)) # size=(17, 64, 358), stride=(1088, 64, 1), dtype=float32, device=cuda
arg2 = tf.Variable(tf.random.uniform([261, 1, 64], dtype=tf.float32)) # size=(261, 1, 64), stride=(16704, 1, 64), dtype=float32, device=cuda

if __name__ == '__main__':
    # Eager execution
    with tf.GradientTape() as tape:
        out_eager = foo(arg0, arg1, arg2)
        # Conversion: .sum().backward() -> reduce_sum + tape.gradient
        loss = tf.reduce_sum(out_eager)
    grads = tape.gradient(loss, [arg0, arg1, arg2])
    print('Eager Success! ✅')
    
    # Conversion: torch.compile -> tf.function
    compiled_foo = tf.function(foo)
    with tf.GradientTape() as tape:
        out_compiled = compiled_foo(arg0, arg1, arg2)
        loss = tf.reduce_sum(out_compiled)
    grads = tape.gradient(loss, [arg0, arg1, arg2])
    print('Compile Success! ✅')
```