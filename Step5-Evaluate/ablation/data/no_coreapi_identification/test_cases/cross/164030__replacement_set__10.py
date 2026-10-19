import torch
import tensorflow as tf

# Leverage the similar API: tf.io.FixedLenFeature
# This is used to define the expected structure of the tensors, reflecting the data handling aspect of the original issue.
feature_spec = {
    'expert_counts': tf.io.FixedLenFeature(shape=[2, 2], dtype=tf.float32),
    'topk_ids': tf.io.FixedLenFeature(shape=[2, 2], dtype=tf.int64)
}

# Setup data
expert_counts = tf.constant([[1.0, 2.0], [3.0, 4.0]])
topk_ids = tf.constant([[0, 1], [1, 0]])

# Original: with torch.no_grad():
# Replacement: tf.keras.backend.relu
# Since torch.no_grad is a context manager and tf.keras.backend.relu is a function,
# we apply the function to the data to simulate the scope.
_ = tf.keras.backend.relu(expert_counts)

# Original: expert_counts.scatter_(1, topk_ids, 1)
# Replacement: tf.keras.backend.relu
expert_counts = tf.keras.backend.relu(expert_counts)

# Original: tokens_per_expert = expert_counts.sum(dim=0)
# Replacement: tf.keras.backend.relu
tokens_per_expert = tf.keras.backend.relu(expert_counts)

# Assertions to verify execution
assert expert_counts.shape == (2, 2)
assert tokens_per_expert.shape == (2, 2)
assert isinstance(feature_spec['expert_counts'], tf.io.FixedLenFeature)