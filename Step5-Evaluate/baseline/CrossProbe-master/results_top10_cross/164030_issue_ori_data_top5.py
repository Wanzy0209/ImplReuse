```python
import tensorflow as tf

# Note: torch.no_grad() is not directly translatable as a context manager in TensorFlow 2.x.
# Gradients are tracked explicitly via tf.GradientTape. If this code is inside a tape
# and gradients should be ignored, wrap inputs in tf.stop_gradient.

# Conversion: expert_counts.scatter_(1, topk_ids, 1)
# PyTorch's scatter_ with dim=1 adds values to specific indices along the columns.
# TensorFlow uses tf.tensor_scatter_nd_add. We need to construct (batch, expert) indices.
batch_indices = tf.range(tf.shape(topk_ids)[0])
batch_indices = tf.tile(tf.expand_dims(batch_indices, 1), [1, tf.shape(topk_ids)[1]])
indices = tf.stack([batch_indices, topk_ids], axis=-1)

# Create updates tensor filled with 1s, matching the dtype of expert_counts
updates = tf.ones_like(topk_ids, dtype=expert_counts.dtype)

# Perform the scatter add operation (returns a new tensor, as TF tensors are immutable)
expert_counts = tf.tensor_scatter_nd_add(expert_counts, indices, updates)

# Conversion: tokens_per_expert = expert_counts.sum(dim=0)
tokens_per_expert = tf.reduce_sum(expert_counts, axis=0)
```