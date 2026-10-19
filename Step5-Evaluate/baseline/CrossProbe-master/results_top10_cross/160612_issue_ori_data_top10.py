```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

# 1. Create Linear layer (Dense in TensorFlow)
# nn.Linear(5, 7) maps to Dense layer with 7 units and input shape (5,)
layer = tf.keras.layers.Dense(7, input_shape=(5,))

# 2. Apply Pruning
# PyTorch: prune.random_unstructured (prunes random units)
# TensorFlow: prune_low_magnitude (prunes low magnitude units by default)
# We use ConstantSparsity to apply 20% pruning immediately (begin_step=0).
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.2, begin_step=0, frequency=1)
}
m = tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)

# Run a dummy step to trigger the pruning update
# (TF pruning applies on batch begin/end, so we need a forward pass)
_ = m(np.ones((1, 5)))

# 3. Remove Pruning
# PyTorch: prune.remove (removes mask/orig buffers, makes pruning permanent)
# TensorFlow: strip_pruning (removes pruning wrappers, makes pruning permanent)
m = tfmot.sparsity.keras.strip_pruning(m)

m
```