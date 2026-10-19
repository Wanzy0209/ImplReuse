```python
import tensorflow as tf
import functools

# Conversion Note: PyTorch's CheckpointPolicy and selective checkpointing contexts
# do not have a direct 1:1 equivalent in TensorFlow's public API.
# TensorFlow uses the @tf.recompute_grad decorator for activation checkpointing.
# The CustomPolicy returning MUST_SAVE implies standard recompute behavior.
#
# class CustomPolicy:
#     def __init__(self):
#         super().__init__()
#
#     def __call__(self, ctx, out, func, *args, **kwargs):
#         return CheckpointPolicy.MUST_SAVE

# from torch.utils.checkpoint import (
#     CheckpointPolicy,
#     create_selective_checkpoint_contexts,
# )

# Conversion Note: @tf.recompute_grad is applied to the function to be checkpointed.
# This replaces the need for torch.utils.checkpoint.checkpoint inside the calling function.
@tf.recompute_grad
def f(x, y):
    # torch.sigmoid -> tf.math.sigmoid
    # torch.matmul -> tf.matmul
    return tf.math.sigmoid(tf.matmul(tf.matmul(x, y), y)) * y

# Conversion Note: context_fn1 and context_fn2 are specific to PyTorch's selective checkpointing
# and are not needed in TensorFlow when using @tf.recompute_grad.
# context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())
#
# def context_fn2():
#     return create_selective_checkpoint_contexts(CustomPolicy())

# Conversion Note: torch.compile is equivalent to tf.function.
# The backend argument is abstracted away by TF's compiler stack (XLA/AutoGraph).
@tf.function(jit_compile=True)
def g(x, y):
    # The checkpointing logic is handled by the decorator on f.
    # use_reentrant=False is the default/safe mode in TF's recompute_grad.
    return f(x, y)

# Conversion Note: torch.randn with requires_grad=True maps to tf.Variable.
# tf.random.normal generates the values.
a = tf.Variable(tf.random.normal((4, 4)), trainable=True)
b = tf.Variable(tf.random.normal((4, 4)), trainable=True)

# Conversion Note: PyTorch's autograd (.backward()) is handled by tf.GradientTape.
with tf.GradientTape() as tape:
    result = g(a, b)
    loss = tf.reduce_sum(result)

# Compute gradients
gradients = tape.gradient(loss, [a, b])
```