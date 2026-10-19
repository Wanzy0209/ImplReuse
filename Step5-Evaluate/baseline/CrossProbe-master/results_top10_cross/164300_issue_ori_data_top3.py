```python
import tensorflow as tf
import functools

# Conversion Note: PyTorch's CheckpointPolicy is specific to torch.utils.checkpoint.
# TensorFlow does not have an equivalent enum for selective checkpointing policies.
# class CheckpointPolicy:
#     MUST_SAVE = 0

class CustomPolicy:
    def __init__(self):
        super().__init__()

    def __call__(self, ctx, out, func, *args, **kwargs):
        # PyTorch: return CheckpointPolicy.MUST_SAVE
        # TF equivalent: This logic is handled internally by tf.recompute_grad
        pass

# Conversion Note: create_selective_checkpoint_contexts is PyTorch specific.
# In TF, we use the @tf.recompute_grad decorator on the function to be checkpointed.
def create_selective_checkpoint_contexts(policy):
    # No-op in TF translation
    return None

def f(x, y):
    # torch.matmul -> tf.linalg.matmul
    # torch.sigmoid -> tf.math.sigmoid
    return tf.math.sigmoid(tf.linalg.matmul(tf.linalg.matmul(x, y), y)) * y

# Apply checkpointing to function f
# Note: tf.recompute_grad is available in TensorFlow 2.x
try:
    recompute_grad = tf.recompute_grad
except AttributeError:
    from tensorflow.python.ops.recompute_grad import recompute_grad

f_checkpointed = recompute_grad(f)

# Conversion Note: torch.compile -> tf.function
@tf.function
def g(x, y):
    # PyTorch calls checkpoint dynamically. In TF, we call the decorated function.
    # The context_fn logic (selective saving) is abstracted away by the TF implementation.
    return f_checkpointed(x, y)

# torch.randn -> tf.random.normal
# requires_grad=True -> handled by tf.GradientTape
a = tf.random.normal((4, 4))
b = tf.random.normal((4, 4))

# Execution and backward pass
with tf.GradientTape() as tape:
    tape.watch([a, b])
    result = g(a, b)
    loss = tf.reduce_sum(result)

grads = tape.gradient(loss, [a, b])
```