```python
import tensorflow as tf

# Conversion: torch.nested.nested_tensor with layout=torch.jagged maps to tf.RaggedTensor
# We use tf.ragged.constant to construct the ragged tensor from a list of sequences
# Note: tf.ragged.constant expects a nested list structure, so we use list(range(n))
x = tf.ragged.constant(
    [list(range(n)) for n in (10, 20, 30)]
)

# Conversion: x.max(dim=1).values maps to tf.reduce_max(x, axis=1)
# tf.reduce_max on a RaggedTensor returns a dense Tensor of the maximum values
print(tf.reduce_max(x, axis=1))
```