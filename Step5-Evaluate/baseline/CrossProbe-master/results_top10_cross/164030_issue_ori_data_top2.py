```python
import tensorflow as tf

# Note: TensorFlow 2.x eager execution does not track gradients automatically,
# so the torch.no_grad() context is not required.

# expert_counts.scatter_(1, topk_ids, 1)
# Construct indices for tf.tensor_scatter_nd_add
# We need to map (batch_idx, expert_idx) -> update
batch_size = tf.shape(topk_ids)[0]
k = tf.shape(topk_ids)[1]

# Create row indices [0, 0, ..., 1, 1, ...] corresponding to each element in topk_ids
row_indices = tf.repeat(tf.range(batch_size), k)
# Flatten topk_ids to match the row indices
col_indices = tf.reshape(topk_ids, [-1])
# Combine into a single index tensor of shape [batch_size * k, 2]
indices = tf.stack([row_indices, col_indices], axis=1)

# Create updates (all 1s) matching the dtype of expert_counts
updates = tf.ones_like(col_indices, dtype=expert_counts.dtype)

# Perform the scatter add operation
expert_counts = tf.tensor_scatter_nd_add(expert_counts, indices, updates)

# tokens_per_expert = expert_counts.sum(dim=0)
tokens_per_expert = tf.reduce_sum(expert_counts, axis=0)
```