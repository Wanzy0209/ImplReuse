```python
import tensorflow as tf

# Conversion Note: torch.no_grad() disables gradient calculation. 
# In TensorFlow 2.x, gradients are only tracked inside tf.GradientTape. 
# If this code runs outside a tape, no gradients are tracked. 
# If inside a tape and gradients should be ignored, wrap inputs in tf.stop_gradient().

# Prepare indices for scatter operation
# PyTorch scatter_(dim, index, value) -> TF tensor_scatter_nd_update(tensor, indices, updates)
# We need to construct the (batch, index) pairs for TF

# Assuming expert_counts shape is (batch_size, num_experts)
# and topk_ids shape is (batch_size, k)
batch_size = tf.shape(topk_ids)[0]
k = tf.shape(topk_ids)[1]

# Create row indices [0, 0, ..., 1, 1, ...] corresponding to the batch dimension
batch_indices = tf.range(batch_size)
batch_indices = tf.repeat(batch_indices, k)

# Flatten topk_ids to get the column indices
expert_indices = tf.reshape(topk_ids, [-1])

# Stack to form (N, 2) indices required by TF
indices = tf.stack([batch_indices, expert_indices], axis=1)

# Create updates (all 1s) matching the number of indices
updates = tf.ones_like(expert_indices, dtype=expert_counts.dtype)

# Perform the scatter update (equivalent to in-place scatter_ with overwrite)
# Note: PyTorch scatter_ overwrites. If accumulation was intended, use tensor_scatter_nd_add.
expert_counts = tf.tensor_scatter_nd_update(expert_counts, indices, updates)

# Sum along dimension 0
tokens_per_expert = tf.reduce_sum(expert_counts, axis=0)
```