```python
import tensorflow as tf
import functools

# Conversion Note: PyTorch's CheckpointPolicy and selective contexts do not have direct equivalents in TensorFlow.
# TensorFlow uses tf.recompute_grad which recomputes the entire function's internals.
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
# Conversion Note: create_selective_checkpoint_contexts is not available in TensorFlow.

# Conversion Note: Applying tf.recompute_grad to function f to simulate torch.utils.checkpoint.checkpoint
@tf.recompute_grad
def f(x, y):
    # torch.sigmoid -> tf.keras.activations.sigmoid (per provided context)
    # torch.matmul -> tf.linalg.matmul
    return tf.keras.activations.sigmoid(tf.linalg.matmul(tf.linalg.matmul(x, y), y)) * y

# Conversion Note: context_fn1 and context_fn2 are PyTorch specific for selective checkpointing.
# In TF, the recompute policy is defined by the decorator on the function.
# context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())
#
# def context_fn2():
#     return create_selective_checkpoint_contexts(CustomPolicy())

# Conversion Note: torch.compile is replaced by tf.function
@tf.function
def g(x, y):
    # torch.utils.checkpoint.checkpoint is handled by @tf.recompute_grad on f
    return f(x, y)

# Conversion Note: torch.randn -> tf.random.normal
# Conversion Note: requires_grad=True implies using tf.Variable
a = tf.Variable(tf.random.normal((4, 4)))
b = tf.Variable(tf.random.normal((4, 4)))

# Conversion Note: .backward() is replaced by tf.GradientTape
with tf.GradientTape() as tape:
    result = g(a, b)
    loss = tf.reduce_sum(result)

grads = tape.gradient(loss, [a, b])
```