```python
import tensorflow as tf
import numpy as np

# Conversion: nn.Linear(5, 7) -> tf.keras.layers.Dense(7, input_shape=(5,))
# Note: PyTorch initializes weights immediately, Keras requires a build step or input call
m = tf.keras.layers.Dense(7, input_shape=(5,))

# Initialize weights to allow manipulation
_ = m(tf.ones((1, 5)))

# Conversion: torch.nn.utils.prune.random_unstructured
# Logic: Randomly select 20% of weights to zero out
weights = m.get_weights()
kernel = weights[0] # Corresponds to 'weight' in PyTorch

# Create a random mask
num_elements = kernel.size
num_prune = int(num_elements * 0.2)
prune_indices = np.random.choice(num_elements, num_prune, replace=False)
mask = np.ones(num_elements)
mask[prune_indices] = 0
mask = mask.reshape(kernel.shape)

# Apply the mask
weights[0] = kernel * mask

# Conversion: torch.nn.utils.prune.remove
# Logic: Make the pruning permanent by setting the modified weights back to the layer
m.set_weights(weights)

m
```