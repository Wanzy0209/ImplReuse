```python
import tensorflow as tf
import functools

# Conversion: PyTorch CheckpointPolicy is not directly available in TensorFlow.
# Defined here to preserve code structure and validity.
class CheckpointPolicy:
    MUST_SAVE = True

class CustomPolicy:
    def __init__(self):
        super().__init__()

    def __call__(self, ctx, out, func, *args, **kwargs):
        # Conversion: In TensorFlow, recompute_grad handles the policy internally.
        # Returning True to mimic MUST_SAVE behavior.
        return CheckpointPolicy.MUST_SAVE

# Conversion: PyTorch create_selective_checkpoint_contexts is not directly available.
# Defined here to preserve code structure.
def create_selective_checkpoint_contexts(policy):
    return None

def f(x, y):
    # Conversion: torch.matmul -> tf.matmul, torch.sigmoid -> tf.math.sigmoid
    return tf.math.sigmoid(tf.matmul(tf.matmul(x, y), y)) * y

context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())

def context_fn2():
    return create_selective_checkpoint_contexts(CustomPolicy())

# Conversion: torch.compile -> tf.function with jit_compile=True
# Conversion: torch.utils.checkpoint.checkpoint -> tf.recompute_grad
# Note: tf.recompute_grad is a decorator, so we apply it to f to mimic the checkpointing behavior.
# The context_fn and use_reentrant arguments are specific to PyTorch and not supported in TF.
@tf.function(jit_compile=True)
def g(x, y):
    # We wrap f with recompute_grad to enable activation checkpointing
    f_checkpointed = tf.recompute_grad(f)
    return f_checkpointed(x, y)

# Conversion: torch.randn -> tf.random.normal
# Conversion: requires_grad=True -> handled by tf.GradientTape watching
a = tf.random.normal((4, 4))
b = tf.random.normal((4, 4))

# Conversion: .backward() -> tf.GradientTape
with tf.GradientTape() as tape:
    tape.watch([a, b])
    result = g(a, b)
    loss = tf.reduce_sum(result)

gradients = tape.gradient(loss, [a, b])
```