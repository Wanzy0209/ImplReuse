```python
import tensorflow as tf

# Conversion: torch.no_grad is not required in TensorFlow eager execution 
# unless explicitly inside a tf.GradientTape. Gradients are tracked opt-in.
# We assume expert_counts is a tf.Variable to support in-place updates.

# Conversion: torch.scatter_(dim, index, src) -> tf.tensor_scatter_nd_add
# We need to construct the (batch, expert) index pairs for TensorFlow.
batch_size = tf.shape(topk_ids)[0]
num_selected = tf.shape(topk_ids)[1]

# Create batch indices [0, 0, ..., 1, 1, ...]
batch_indices = tf.range(batch_size)
batch_indices = tf.repeat(batch_indices, num_selected)
batch_indices = tf.expand_dims(batch_indices, axis=1)

# Flatten expert indices from topk_ids
expert_indices = tf.reshape(topk_ids, [-1])
expert_indices = tf.expand_dims(expert_indices, axis=1)

# Combine to form [[batch_idx, expert_idx], ...]
indices = tf.concat([batch_indices, expert_indices], axis=1)

# Create updates (all 1s, matching the number of indices)
updates = tf.ones_like(expert_indices, dtype=expert_counts.dtype)

# Perform the scatter add operation (in-place update for Variable)
expert_counts.assign(tf.tensor_scatter_nd_add(expert_counts, indices, updates))

# Conversion: tensor.sum(dim=0) -> tf.reduce_sum(tensor, axis=0)
tokens_per_expert = tf.reduce_sum(expert_counts, axis=0)
```