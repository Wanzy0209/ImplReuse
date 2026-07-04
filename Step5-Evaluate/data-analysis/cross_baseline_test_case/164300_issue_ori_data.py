```python
import tensorflow as tf
import functools

# Conversion: PyTorch's CheckpointPolicy and custom policies are specific to its checkpointing implementation.
# TensorFlow uses tf.recompute_grad which handles the policy of recomputing the forward pass internally.
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

def f(x, y):
    # Conversion: torch.matmul -> tf.linalg.matmul
    # Conversion: torch.sigmoid -> tf.math.sigmoid
    return tf.math.sigmoid(tf.linalg.matmul(tf.linalg.matmul(x, y), y)) * y

# Conversion: context_fn1 and context_fn2 are PyTorch specific mechanisms for selective checkpointing.
# In TensorFlow, selective checkpointing is handled differently, often by splitting functions or using tf.recompute_grad.
# context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())
#
# def context_fn2():
#     return create_selective_checkpoint_contexts(CustomPolicy())

# Conversion: torch.compile -> tf.function
# Conversion: torch.utils.checkpoint.checkpoint -> tf.recompute_grad
# We apply tf.recompute_grad to function f to achieve the memory saving effect.
@tf.recompute_grad
def f_recomputed(x, y):
    return f(x, y)

@tf.function
def g(x, y):
    # The PyTorch code calls checkpoint(f, ...) inside g.
    # In TF, we call the recompute-decorated version of f.
    return f_recomputed(x, y)

# Conversion: torch.randn -> tf.random.normal
# Conversion: requires_grad=True -> Variables watched by GradientTape
a = tf.random.normal((4, 4))
b = tf.random.normal((4, 4))

# Conversion: g(a, b).sum().backward() -> GradientTape execution
with tf.GradientTape() as tape:
    tape.watch(a)
    tape.watch(b)
    output = g(a, b)
    loss = tf.reduce_sum(output)

gradients = tape.gradient(loss, [a, b])
```