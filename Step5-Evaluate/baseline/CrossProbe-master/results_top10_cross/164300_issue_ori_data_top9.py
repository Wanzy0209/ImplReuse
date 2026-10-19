```python
import tensorflow as tf
import functools
# from torch.utils.checkpoint import CheckpointPolicy # No direct TF equivalent

# Conversion: CustomPolicy is specific to PyTorch's selective checkpointing.
# TensorFlow's recompute_grad does not support custom runtime policies.
# class CustomPolicy:
#     def __init__(self):
#         super().__init__()
#     def __call__(self, ctx, out, func, *args, **kwargs):
#         return CheckpointPolicy.MUST_SAVE

# from torch.utils.checkpoint import ... # No direct TF equivalent

# Conversion: torch.utils.checkpoint.checkpoint is equivalent to @tf.recompute_grad
@tf.recompute_grad
def f(x, y):
    # Conversion: torch.sigmoid -> tf.nn.sigmoid, torch.matmul -> tf.matmul
    return tf.nn.sigmoid(tf.matmul(tf.matmul(x, y), y)) * y

# Conversion: create_selective_checkpoint_contexts is not available in TF.
# context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())
# def context_fn2():
#     return create_selective_checkpoint_contexts(CustomPolicy())

# Conversion: torch.compile -> tf.function
@tf.function(jit_compile=True)
def g(x, y):
    # In PyTorch, checkpointing is applied here. In TF, it is applied via decorator on f.
    # use_reentrant=False and context_fn are specific to PyTorch.
    return f(x, y)

# Conversion: torch.randn -> tf.random.normal
# Conversion: requires_grad=True -> tf.Variable
a = tf.Variable(tf.random.normal((4, 4)))
b = tf.Variable(tf.random.normal((4, 4)))

# Conversion: .backward() -> tf.GradientTape
with tf.GradientTape() as tape:
    result = g(a, b)
    loss = tf.reduce_sum(result)

grads = tape.gradient(loss, [a, b])
```