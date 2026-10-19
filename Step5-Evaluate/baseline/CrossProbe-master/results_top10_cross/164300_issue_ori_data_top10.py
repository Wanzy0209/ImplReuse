```python
import tensorflow as tf
# import functools # Not strictly needed in TF translation as context_fn logic differs

# PyTorch's CheckpointPolicy and selective contexts don't have a direct 1:1 equivalent in TensorFlow.
# TensorFlow uses tf.recompute_grad for activation checkpointing.
# We keep the class structure to match the source, but it is not used in the TF logic.
class CustomPolicy:
    def __init__(self):
        super().__init__()

    def __call__(self, ctx, out, func, *args, **kwargs):
        # PyTorch: return CheckpointPolicy.MUST_SAVE
        # TensorFlow: tf.recompute_grad handles the decision to recompute internally.
        pass

# PyTorch: from torch.utils.checkpoint import ...
# TensorFlow: No direct import for these specific utilities.

def f(x, y):
    # Conversion: torch.sigmoid -> tf.math.sigmoid
    # Conversion: torch.matmul -> tf.matmul
    return tf.math.sigmoid(tf.matmul(tf.matmul(x, y), y)) * y

# Conversion: functools.partial(create_selective_checkpoint_contexts, CustomPolicy())
# TensorFlow: tf.recompute_grad is used to wrap the function for checkpointing.
# We apply it here to mimic the behavior of the context function.
f_recomputed = tf.recompute_grad(f)

# Conversion: def context_fn2(): ...
# TensorFlow: Not applicable, handled by tf.recompute_grad.
def context_fn2():
    pass

# Conversion: @torch.compile(backend="aot_eager_decomp_partition", fullgraph=True)
# TensorFlow: @tf.function is the standard decorator to compile a graph in TensorFlow.
@tf.function
def g(x, y):
    # Conversion: torch.utils.checkpoint.checkpoint(f, x, y, use_reentrant=False, context_fn=context_fn1)
    # TensorFlow: Call the function wrapped with tf.recompute_grad.
    return f_recomputed(x, y)

# Conversion: torch.randn(4, 4, requires_grad=True, device="cpu")
# TensorFlow: tf.random.normal. Gradients are tracked via GradientTape.
a = tf.random.normal((4, 4))
b = tf.random.normal((4, 4))

# Conversion: g(a, b).sum().backward()
# TensorFlow: Use GradientTape to compute gradients.
with tf.GradientTape() as tape:
    tape.watch([a, b])
    result = g(a, b)
    loss = tf.reduce_sum(result)

grads = tape.gradient(loss, [a, b])
```