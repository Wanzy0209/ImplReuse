```python
import tensorflow as tf
import tensorflow_model_optimization as tfmot

# PyTorch: prune.random_unstructured(nn.Linear(5, 7), name="weight", amount=0.2)
# TensorFlow: prune_low_magnitude wraps the layer. ConstantSparsity mimics the immediate random pruning.
# Note: TF pruning applies to the kernel (weights) by default, corresponding to "weight" in PyTorch.
m = tfmot.sparsity.keras.prune_low_magnitude(
    tf.keras.layers.Dense(7, input_shape=(5,)),
    pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(0.2, begin_step=0, frequency=1)
)

# PyTorch: prune.remove(m, name="weight")
# TensorFlow: strip_pruning makes the pruning permanent by removing the wrapper and keeping pruned weights.
# We call the layer to ensure it is built and the mask is applied before stripping.
_ = m(tf.ones((1, 5)))
m = tfmot.sparsity.keras.strip_pruning(m)

m
```