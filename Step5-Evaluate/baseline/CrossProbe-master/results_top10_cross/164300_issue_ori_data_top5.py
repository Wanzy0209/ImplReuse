```python
import tensorflow as tf
import functools

# Conversion Note: PyTorch's CheckpointPolicy and selective checkpointing contexts
# are specific to PyTorch. TensorFlow uses the @tf.recompute_grad decorator.
# The following class is preserved for structural compatibility but is a no-op.
class CustomPolicy:
    def __init__(self):
        super().__init__()

    def __call__(self, ctx, out, func, *args, **kwargs):
        # PyTorch: return CheckpointPolicy.MUST_SAVE
        # TF: No direct equivalent for selective policy in recompute_grad
        pass

# Conversion Note: create_selective_checkpoint_contexts is PyTorch specific.
# Defined as a placeholder to preserve structure.
def create_selective_checkpoint_contexts(policy):
    return None

# Conversion Note: torch.utils.checkpoint.checkpoint is replaced by @tf.recompute_grad.
# This decorator recomputes the forward pass during backprop to save memory.
@tf.recompute_grad
def f(x, y):
    # torch.matmul -> tf.matmul
    # torch.sigmoid -> tf.math.sigmoid
    return tf.math.sigmoid(tf.matmul(tf.matmul(x, y), y)) * y

# Preserving structure of context_fn definitions
context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())

def context_fn2():
    return create_selective_checkpoint_contexts(CustomPolicy())

# Conversion Note: torch.compile is replaced by @tf.function.
# jit_compile=True is used to approximate the compilation aspect.
@tf.function(jit_compile=True)
def g(x, y):
    # In the PyTorch code, checkpointing is applied here with a context function.
    # In TensorFlow, checkpointing is handled by the decorator on f.
    # We simply call f(x, y).
    return f(x, y)

# Conversion Note: torch.randn -> tf.random.normal
# requires_grad is handled implicitly by tf.GradientTape
a = tf.random.normal((4, 4))
b = tf.random.normal((4, 4))

# Conversion Note: PyTorch's .backward() is replaced by tf.GradientTape.
with tf.GradientTape() as tape:
    result = g(a, b)
    loss = tf.reduce_sum(result)

gradients = tape.gradient(loss, [a, b])
```