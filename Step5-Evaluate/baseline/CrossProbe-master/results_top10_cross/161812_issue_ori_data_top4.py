```python
import tensorflow as tf

# Conversion: torch.nested.nested_tensor with layout=th.jagged maps to tf.RaggedTensor
x = tf.ragged.constant([tf.ones((3, 2, 3)), tf.ones((4, 2, 3))])

# Conversion: torch.cat maps to tf.concat
tf.concat([x, x], axis=0)
```