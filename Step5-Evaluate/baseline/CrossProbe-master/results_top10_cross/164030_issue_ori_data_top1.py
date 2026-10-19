```python
import tensorflow as tf

# Note: TensorFlow does not track gradients automatically in eager execution.
# To mimic torch.no_grad() behavior inside a tf.GradientTape, wrap the result in tf.stop_gradient().

# expert_counts.scatter_(1, topk_ids, 1)
# PyTorch scatter_ is in-place. TF tensors are immutable, so we use tensor_scatter_nd_add.
# We need to construct indices for the scatter operation.

# Assuming shapes: expert_counts=[batch_size, num_experts], topk_ids=[batch_size, top_k]
batch_size = tf.shape(expert_counts)[0]

# Create row indices [0, 0, ..., 1, 1, ...] matching the shape of topk_ids
row_indices = tf.range(batch_size)
row_indices = tf.expand_dims(row_indices, 1)
row_indices = tf.tile(row_indices, [1, tf.shape(topk_ids)[1]])

# Stack row and column indices to get [batch_size, top_k, 2]
indices = tf.stack([row_indices, topk_ids], axis=-1)

# Reshape to [batch_size * top_k, 2] for scatter_nd
indices = tf.reshape(indices, [-1, 2])

# Create updates (all 1s) and reshape to [batch_size * top_k]
updates = tf.ones_like(topk_ids, dtype=expert_counts.dtype)
updates = tf.reshape(updates, [-1])

# Perform scatter add
expert_counts = tf.tensor_scatter_nd_add(expert_counts, indices, updates)

# tokens_per_expert = expert_counts.sum(dim=0)
tokens_per_expert = tf.reduce_sum(expert_counts, axis=0)
```