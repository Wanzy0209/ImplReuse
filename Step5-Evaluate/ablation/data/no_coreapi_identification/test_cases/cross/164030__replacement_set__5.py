import torch
import tensorflow as tf

# Leverage the similar API: tf.io.FixedLenFeature
# This mirrors the configuration aspect of the bug report (defining shapes/types).
# We use it to define the expected shape of our tensors.
feature_shape = [2, 2]
feature_spec = tf.io.FixedLenFeature(feature_shape, tf.int64)

# Setup data
expert_counts = tf.constant([[1, 2], [3, 4]], dtype=tf.int64)
topk_ids = tf.constant([[0, 1], [1, 0]], dtype=tf.int64)

# Original logic: with torch.no_grad(): ...
# Replacement: torch.no_grad -> tf.keras.ops.add
# Note: tf.keras.ops.add is a function, not a context manager.
# We execute the logic block directly.

# Original: expert_counts.scatter_(1, topk_ids, 1)
# Replacement: expert_counts.scatter_ -> tf.keras.ops.add
# We replace the scatter operation with an add operation.
# Arguments are adapted to fit add(x, y).
result_1 = tf.keras.ops.add(expert_counts, topk_ids)

# Original: tokens_per_expert = expert_counts.sum(dim=0)
# Replacement: expert_counts.sum -> tf.keras.ops.add
# We replace the sum operation with an add operation.
# Arguments are adapted to fit add(x, y).
result_2 = tf.keras.ops.add(result_1, 0)

# Verify the output matches the expected shape defined by FixedLenFeature
assert result_2.shape == tf.TensorShape(feature_shape)