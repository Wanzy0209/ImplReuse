```python
import tensorflow as tf

# Create a Dense layer (equivalent to nn.Linear(5, 7))
# Note: In Keras, the first argument is the number of output units (7).
# Input shape (5) is specified to build the layer and initialize weights immediately.
m = tf.keras.layers.Dense(7, input_shape=(5,))

# Initialize weights by calling the layer once (necessary to access kernel)
_ = m(tf.zeros((1, 5)))

# Apply random unstructured pruning (amount=0.2)
# Get the current weights (kernel)
weights = m.kernel
# Generate a random mask where 20% of elements are 0 and 80% are 1
mask = tf.cast(tf.random.uniform(weights.shape) > 0.2, weights.dtype)
# Apply the mask to the weights
pruned_weights = weights * mask

# Remove the pruning reparameterization
# In PyTorch, prune.remove makes the pruning permanent by removing the mask buffer
# and applying the mask to the original parameter.
# In TensorFlow, we assign the pruned values directly to the kernel variable.
m.kernel.assign(pruned_weights)

m
```