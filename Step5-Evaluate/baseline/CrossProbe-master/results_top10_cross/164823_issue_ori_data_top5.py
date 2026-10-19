```python
import tensorflow as tf

# Conversion: nn.Module -> tf.keras.Model
class TestModel(tf.keras.Model):
    # Conversion: forward -> call
    def call(self, x):
        # Conversion: x.to_sparse() -> tf.sparse.from_dense(x)
        x_sparse = tf.sparse.from_dense(x)
        
        # Conversion: Sparse tensor arithmetic requires explicit ops in TF
        # PyTorch: x_sparse * 2
        result = tf.sparse.multiply(x_sparse, 2)
        
        # Conversion: result.to_dense() -> tf.sparse.to_dense(result)
        return tf.sparse.to_dense(result)

# Conversion: torch.randn -> tf.random.normal
x = tf.random.normal((10, 10))

model = TestModel()
print("Eager output:", model(x))

# Conversion: torch.compile -> tf.function
print("Compiled output:", tf.function(model)(x))
```