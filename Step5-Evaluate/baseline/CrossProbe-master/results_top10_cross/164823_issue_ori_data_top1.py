```python
import tensorflow as tf

# Conversion: nn.Module -> tf.Module
class TestModel(tf.Module):
    # Conversion: forward -> __call__ to maintain callable object structure
    def __call__(self, x):
        # Conversion: x.to_sparse() -> tf.sparse.from_dense(x)
        x_sparse = tf.sparse.from_dense(x)
        # Conversion: Sparse tensor multiplication -> tf.sparse.multiply
        result = tf.sparse.multiply(x_sparse, 2)
        # Conversion: result.to_dense() -> tf.sparse.to_dense(result)
        return tf.sparse.to_dense(result)

# Conversion: torch.randn -> tf.random.normal
x = tf.random.normal((10, 10))

model = TestModel()
print("Eager output:", model(x))

# Conversion: torch.compile -> tf.function
# tf.function creates a static graph from the Python function, analogous to compilation in PyTorch
print("Compiled output:", tf.function(model)(x))
```