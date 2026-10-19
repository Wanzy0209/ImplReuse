```python
import tensorflow as tf

# PyTorch: m = prune.random_unstructured(nn.Linear(5, 7), name = "weight", amount= 0.2)
# TensorFlow does not have a direct utility for pruning existing layers like PyTorch.
# We simulate this by creating a Dense layer and manually zeroing out 20% of the weights.
m = tf.keras.layers.Dense(7, input_shape=(5,))

# Get current weights and bias
weights, bias = m.get_weights()

# Generate a random mask to prune 20% of the weights (amount=0.2)
# PyTorch removes 'amount' fraction, so we keep values where random > 0.2
mask = tf.random.uniform(tf.shape(weights)) > 0.2

# Apply mask to weights
pruned_weights = weights * tf.cast(mask, weights.dtype)

# Set the pruned weights back to the layer
m.set_weights([pruned_weights, bias])

# PyTorch: m = prune.remove(m, name = "weight")
# In PyTorch, this makes the pruning permanent by removing the mask buffer.
# Since we manually updated the weights tensor above, the pruning is already permanent.
# No specific function call is needed in TensorFlow to achieve this state.

m
```