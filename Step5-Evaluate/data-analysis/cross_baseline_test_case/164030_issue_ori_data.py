```python
import tensorflow as tf

# Conversion: torch.no_grad is implicit in TF (gradients only tracked inside tf.GradientTape)
# Conversion: scatter_ requires constructing indices for tf.tensor_scatter_nd_add
batch_size = tf.shape(topk_ids)[0]
top_k = tf.shape(topk_ids)[1]

# Create indices for (batch, expert) pairs
batch_indices = tf.repeat(tf.range(batch_size), top_k)
flat_topk_ids = tf.reshape(topk_ids, [-1])
indices = tf.stack([batch_indices, flat_topk_ids], axis=1)

# Create updates (all 1s)
updates = tf.ones_like(flat_topk_ids, dtype=expert_counts.dtype)

# Perform scatter add (returns new tensor, reassign to expert_counts)
expert_counts = tf.tensor_scatter_nd_add(expert_counts, indices, updates)

tokens_per_expert = tf.reduce_sum(expert_counts, axis=0)
```