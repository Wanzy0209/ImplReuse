```python
import tensorflow as tf
import functools

# Conversion: torch.utils.checkpoint.CheckpointPolicy is PyTorch specific.
# TensorFlow uses tf.recompute_grad for activation checkpointing.
class CustomPolicy:
    def __init__(self):
        super().__init__()

    def __call__(self, ctx, out, func, *args, **kwargs):
        # In PyTorch this returns CheckpointPolicy.MUST_SAVE.
        # In TF, tf.recompute_grad determines the policy automatically (recompute all).
        return True

# Conversion: create_selective_checkpoint_contexts is PyTorch specific.
# We define a placeholder to preserve structure.
def create_selective_checkpoint_contexts(policy):
    # This function is not directly used in TF's gradient tape logic
    return policy

def f(x, y):
    # Conversion: torch.matmul -> tf.linalg.matmul
    # Note: Context suggested tf.keras.layers.core.dense, but that is a layer, 
    # not a functional op. tf.linalg.matmul is the correct functional equivalent.
    # Conversion: torch.sigmoid -> tf.keras.backend.sigmoid
    return tf.keras.backend.sigmoid(tf.linalg.matmul(tf.linalg.matmul(x, y), y)) * y

context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())

def context_fn2():
    return create_selective_checkpoint_contexts(CustomPolicy())

# Conversion: torch.compile -> tf.function
# Conversion: torch.utils.checkpoint.checkpoint -> tf.recompute_grad
# We apply tf.recompute_grad to function f to mimic the checkpointing behavior.
# Note: context_fn logic is handled internally by tf.recompute_grad.
f_checkpointed = tf.recompute_grad(f)

@tf.function
def g(x, y):
    # PyTorch calls checkpoint(f, ...) here.
    # TF calls the recompute_grad wrapped function.
    return f_checkpointed(x, y)

# Conversion: torch.randn -> tf.random.normal
# Conversion: requires_grad=True -> trainable=True in tf.Variable
a = tf.Variable(tf.random.normal((4, 4)), trainable=True)
b = tf.Variable(tf.random.normal((4, 4)), trainable=True)

# Conversion: g(a, b).sum().backward()
with tf.GradientTape() as tape:
    result = g(a, b)
    loss = tf.reduce_sum(result)

grads = tape.gradient(loss, [a, b])
```