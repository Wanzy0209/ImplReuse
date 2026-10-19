```python
import tensorflow as tf

# Conversion: nn.Module -> tf.Module
class TestModel(tf.Module):
    # Conversion: forward -> __call__
    def __call__(self, x):
        # Conversion: x.to_sparse() -> tf.sparse.from_dense
        x_sparse = tf.sparse.from_dense(x)
        # Sparse arithmetic is supported directly in TF2
        result = x_sparse * 2
        # Conversion: result.to_dense() -> tf.sparse.to_dense
        return tf.sparse.to_dense(result)

# Conversion: torch.randn -> tf.random.normal
x = tf.random.normal((10, 10))

model = TestModel()
print("Eager output:", model(x))
# Conversion: torch.compile -> tf.function
print("Compiled output:", tf.function(model)(x))
```