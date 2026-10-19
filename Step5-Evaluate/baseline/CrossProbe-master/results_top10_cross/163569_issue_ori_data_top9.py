```python
import tensorflow as tf
import sys

# PyTorch specific configs are omitted as they do not apply to TensorFlow
# torch._dynamo.config.capture_scalar_outputs = True
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
# torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0 # size=(2, 261, 17, 358), dtype=bfloat16
    # Conversion: torch.max(dim=0) -> tf.reduce_max(axis=0)
    t1 = tf.reduce_max(t0, axis=0) # size=(261, 17, 358), dtype=bfloat16
    # Conversion: torch.transpose -> tf.transpose with perm
    t2 = tf.transpose(t1, perm=[1, 0, 2]) # size=(17, 261, 358), dtype=bfloat16
    
    t3 = arg1 # size=(17, 64, 358), dtype=float32
    t4 = tf.exp(t3) # size=(17, 64, 358), dtype=float32
    
    t5 = arg2 # size=(261, 1, 64), dtype=float32
    t6 = tf.transpose(t5, perm=[0, 2, 1]) # size=(261, 64, 1), dtype=float32
    
    # Conversion: torch.nn.functional.conv1d
    # PyTorch input: (N, C_in, L), weight: (C_out, C_in, K)
    # TensorFlow input: (N, L, C_in), filter: (K, C_in, C_out)
    # t4: (17, 64, 358) -> (17, 358, 64)
    # t6: (261, 64, 1) -> (1, 64, 261)
    # Output needs to be (17, 261, 358)
    t4_transposed = tf.transpose(t4, [0, 2, 1])
    t6_transposed = tf.transpose(t6, [2, 1, 0])
    conv_res = tf.nn.conv1d(t4_transposed, t6_transposed, stride=1, padding='VALID')
    t7 = tf.transpose(conv_res, [0, 2, 1]) # size=(17, 261, 358), dtype=float32
    
    # Conversion: t8 = t7.clone(); t8.zero_()
    # Tensors in TF are immutable, so we create a new zero tensor
    t8 = tf.zeros_like(t7) # size=(17, 261, 358), dtype=float32
    
    t9 = t2 * t7 * t8 # size=(17, 261, 358), dtype=float32
    output = t9  # output tensor
    return output

# Conversion: torch.rand -> tf.random.uniform
# Conversion: device='cuda' is handled automatically by TensorFlow if available
arg0 = tf.random.uniform([2, 261, 17, 358], dtype=tf.bfloat16) # size=(2, 261, 17, 358), dtype=bfloat16
arg1 = tf.random.uniform([17, 64, 358], dtype=tf.float32) # size=(17, 64, 358), dtype=float32
arg2 = tf.random.uniform([261, 1, 64], dtype=tf.float32) # size=(261, 1, 64), dtype=float32

if __name__ == '__main__':
    # Eager execution
    with tf.GradientTape() as tape:
        # Conversion: requires_grad=True -> tape.watch
        tape.watch([arg0, arg1, arg2])
        out_eager = foo(arg0, arg1, arg2)
    # Conversion: .backward() -> tape.gradient
    grads = tape.gradient(tf.reduce_sum(out_eager), [arg0, arg1, arg2])
    print('Eager Success! ✅')
    
    # Compiled execution
    # Conversion: torch.compile -> tf.function
    compiled_foo = tf.function(foo)
    with tf.GradientTape() as tape:
        tape.watch([arg0, arg1, arg2])
        out_compiled = compiled_foo(arg0, arg1, arg2)
    grads = tape.gradient(tf.reduce_sum(out_compiled), [arg0, arg1, arg2])
    print('Compile Success! ✅')
```