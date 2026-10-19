```python
import tensorflow as tf

# Conversion: torch.nested.nested_tensor with layout=torch.jagged maps to tf.RaggedTensor.
# We use tf.ragged.constant to construct it from a list of variable-length tensors.
x = tf.ragged.constant([tf.range(0, n) for n in (10, 20, 30)])

# Conversion: x.max(dim=1).values maps to tf.reduce_max(x, axis=1).
# In TF RaggedTensors, axis=1 refers to the inner jagged dimension.
print(tf.reduce_max(x, axis=1))
```