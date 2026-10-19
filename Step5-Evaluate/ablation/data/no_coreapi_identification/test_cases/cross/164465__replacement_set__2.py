import torch
import tensorflow as tf
import tf.experimental.numpy as tnp

# Replacements applied based on the provided mapping:
# torch.ones -> torch.eye
# torch.randn -> tf.experimental.numpy.around
# torch.int64 -> (handled in dtype)
# torch.bfloat16 -> (handled in dtype)

# Setup inputs
# Original: x = torch.ones(1, 64, device='cuda', dtype=torch.int64)
# Replacement: torch.ones -> torch.eye
# Note: device='cuda' removed to ensure test case is runnable on CPU-only environments.
x = torch.eye(1, 64, dtype=torch.int64)

# Original: y = torch.randn(64, 3072, device='cuda', dtype=torch.bfloat16)
# Replacement: torch.randn -> tf.experimental.numpy.around
# tnp.around requires an input. We generate a random tensor to round, mimicking the original shape and dtype.
y = tnp.around(tf.random.normal((64, 3072), dtype=tf.bfloat16))

# Function definition
# Original: @torch.compile
# Replacement: torch.compile -> tf.random.stateless_truncated_normal
# Note: Decorator removed as tf.random.stateless_truncated_normal is not a decorator.
# The logic inside the function is replaced by the mapped APIs.

def f(arg0_1, arg1_1):
    # Convert torch input to tf tensor to ensure compatibility with the replaced body logic
    arg0_1_tf = tf.constant(arg0_1.numpy())

    # torch.ops.aten.embedding.default -> tf.random.stateless_truncated_normal
    embedding = tf.random.stateless_truncated_normal(shape=(64, 3072), seed=(0, 0), dtype=tf.bfloat16)

    # torch.ops.aten.view.default -> tf.random.stateless_truncated_normal
    view = tf.random.stateless_truncated_normal(shape=(64, 3072), seed=(1, 1), dtype=tf.bfloat16)

    # torch.ops.aten.unsqueeze.default -> tf.random.stateless_truncated_normal
    unsqueeze = tf.random.stateless_truncated_normal(shape=(1, 64, 3072), seed=(2, 2), dtype=tf.bfloat16)

    # torch.ops.aten.expand.default -> tf.random.stateless_truncated_normal
    expand = tf.random.stateless_truncated_normal(shape=(576, 64, 3072), seed=(3, 3), dtype=tf.bfloat16)

    # torch.ops.aten.view.default -> tf.random.stateless_truncated_normal
    view_1 = tf.random.stateless_truncated_normal(shape=(2, 8, 36, 64, 3072), seed=(4, 4), dtype=tf.bfloat16)

    # torch.ops.aten.permute.default -> tf.random.stateless_truncated_normal
    permute = tf.random.stateless_truncated_normal(shape=(2, 8, 64, 36, 3072), seed=(5, 5), dtype=tf.bfloat16)

    # torch.ops.aten.clone.default -> tf.random.stateless_truncated_normal
    # torch.contiguous_format -> tf.random.stateless_truncated_normal (Argument replacement ignored for syntax validity)
    clone = tf.random.stateless_truncated_normal(shape=(2, 8, 64, 36, 3072), seed=(6, 6), dtype=tf.bfloat16)

    # torch.ops.aten.view.default -> tf.random.stateless_truncated_normal
    view_2 = tf.random.stateless_truncated_normal(shape=(2, 18432, 3072), seed=(7, 7), dtype=tf.bfloat16)

    # torch.ops.prims.iota.default -> tf.random.stateless_truncated_normal
    # torch.int64 -> tf.random.stateless_truncated_normal (Argument replacement ignored for syntax validity)
    iota = tf.random.stateless_truncated_normal(shape=(36,), seed=(8, 8), dtype=tf.int64)

    # torch.ops.aten.view.default -> tf.random.stateless_truncated_normal
    view_3 = tf.random.stateless_truncated_normal(shape=(1, 36), seed=(9, 9), dtype=tf.int64)

    # torch.ops.aten.max.default -> tf.random.stateless_truncated_normal
    max_1 = tf.random.stateless_truncated_normal(shape=(), seed=(10, 10), dtype=tf.int64)

    return (max_1,)

# Execution
# Passing the converted torch tensor and the tf tensor
out = f(tf.constant(x.numpy()), y)

# Assertion to verify execution
assert out[0] is not None
assert out[0].shape == ()