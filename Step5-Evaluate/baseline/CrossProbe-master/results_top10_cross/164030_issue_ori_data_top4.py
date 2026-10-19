```python
import tensorflow as tf

# In TensorFlow, gradients are tracked explicitly via tf.GradientTape.
# Operations outside of a GradientTape context do not track gradients,
# mimicking the behavior of torch.no_grad().
# If inside a tape, use tf.stop_gradient() on inputs to prevent tracking.

# PyTorch: expert_counts.scatter_(1, topk_ids, 1)
# Construct indices for scatter operation
# topk_ids shape is assumed to be (batch_size, top_k)
batch_size = tf.shape(topk_ids)[0]
top_k = tf.shape(topk_ids)[1]

# Create batch indices (0, 0, ..., 1, 1, ...) corresponding to each top_k element
batch_indices = tf.repeat(tf.range(batch_size), top_k)
# Flatten topk_ids to match the batch_indices
expert_indices = tf.reshape(topk_ids, [-1])
# Stack to form (batch_idx, expert_idx) pairs
indices = tf.stack([batch_indices, expert_indices], axis=1)

# Create updates (all 1s) matching the number of indices
# Ensure dtype matches expert_counts for the scatter operation
updates = tf.ones_like(expert_indices, dtype=expert_counts.dtype)

# Perform scatter add.
# Note: expert_counts must be a tf.Variable for in-place update (scatter_nd_add).
# If expert_counts is a Tensor, use tf.tensor_scatter_nd_add and reassign.
expert_counts.scatter_nd_add(indices, updates)

# PyTorch: tokens_per_expert = expert_counts.sum(dim=0)
tokens_per_expert = tf.reduce_sum(expert_counts, axis=0)
```