```python
import tensorflow as tf

# Conversion: torch.nested.nested_tensor with layout=torch.jagged maps to tf.RaggedTensor.
# torch.arange(0, n) maps to tf.range(n).
x = tf.ragged.constant(
    [tf.range(n) for n in (10, 20, 30)]
)
# Conversion: x.max(dim=1).values maps to tf.reduce_max along axis 1.
# PyTorch returns a named tuple (values, indices), TF returns the tensor directly.
print(tf.reduce_max(x, axis=1))
```