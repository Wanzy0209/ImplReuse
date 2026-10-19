```python
import tensorflow as tf
import functools

# Conversion Comment: CheckpointPolicy is PyTorch specific.
# from torch.utils.checkpoint import CheckpointPolicy

# Conversion Comment: CustomPolicy is PyTorch specific for selective checkpointing.
# TensorFlow does not support custom selective checkpointing policies in the same way.
# class CustomPolicy:
#     def __init__(self):
#         super().__init__()
#
#     def __call__(self, ctx, out, func, *args, **kwargs):
#         return CheckpointPolicy.MUST_SAVE

# Conversion Comment: create_selective_checkpoint_contexts is PyTorch specific.
# from torch.utils.checkpoint import (
#     CheckpointPolicy,
#     create_selective_checkpoint_contexts,
# )

def f(x, y):
    # Conversion Comment: torch.matmul -> tf.matmul, torch.sigmoid -> tf.math.sigmoid
    return tf.math.sigmoid(tf.matmul(tf.matmul(x, y), y)) * y

# Conversion Comment: PyTorch context functions for selective checkpointing are not
# directly translatable. We use tf.recompute_grad for standard activation checkpointing.
# context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())
#
# def context_fn2():
#     return create_selective_checkpoint_contexts(CustomPolicy())

# Conversion Comment: Apply tf.recompute_grad to f to mimic torch.utils.checkpoint.checkpoint
from tensorflow.python.ops.recompute_grad import recompute_grad
f_checkpointed = recompute_grad(f)

# Conversion Comment: torch.compile -> tf.function
@tf.function
def g(x, y):
    # Conversion Comment: torch.utils.checkpoint.checkpoint is replaced by calling the recompute_grad wrapped function.
    # use_reentrant=False is the default behavior in tf.recompute_grad.
    return f_checkpointed(x, y)

# Conversion Comment: torch.randn -> tf.random.normal
# requires_grad=True is handled by tf.GradientTape watching Variables.
a = tf.Variable(tf.random.normal((4, 4)))
b = tf.Variable(tf.random.normal((4, 4)))

# Conversion Comment: .backward() -> tape.gradient()
with tf.GradientTape() as tape:
    result = g(a, b)
    loss = tf.reduce_sum(result)

grads = tape.gradient(loss, [a, b])
```