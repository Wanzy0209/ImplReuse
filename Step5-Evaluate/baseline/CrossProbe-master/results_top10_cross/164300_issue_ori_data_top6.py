```python
import tensorflow as tf
import functools

# Conversion Note: PyTorch's selective checkpointing policies (CustomPolicy) 
# and context creation (create_selective_checkpoint_contexts) are specific 
# to PyTorch's implementation. TensorFlow uses the tf.recompute_grad decorator 
# to achieve activation checkpointing (trading compute for memory).

# class CustomPolicy: ... (Omitted as TF does not support custom policy classes in this way)

@tf.recompute_grad
def f(x, y):
    # torch.sigmoid -> tf.math.sigmoid
    # torch.matmul -> tf.matmul
    return tf.math.sigmoid(tf.matmul(tf.matmul(x, y), y)) * y

# Conversion Note: torch.compile is roughly equivalent to tf.function.
# The specific backend 'aot_eager_decomp_partition' does not have a direct 
# mapping, so we use standard tf.function with JIT compilation.
@tf.function(jit_compile=True)
def g(x, y):
    # In PyTorch, checkpoint is called here. In TF, the recompute logic 
    # is attached to the function definition 'f'.
    return f(x, y)

# torch.randn -> tf.random.normal
# requires_grad=True is implicit in TF when using GradientTape
a = tf.random.normal((4, 4))
b = tf.random.normal((4, 4))

# PyTorch's .backward() is handled by GradientTape in TF
with tf.GradientTape() as tape:
    result = g(a, b)
    loss = tf.reduce_sum(result)

grads = tape.gradient(loss, [a, b])
```