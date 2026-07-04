```python
import tensorflow as tf

class TestModel(tf.keras.Model):
    def call(self, x):
        # Convert dense tensor to sparse tensor
        x_sparse = tf.sparse.from_dense(x)
        # Perform element-wise multiplication on sparse tensor
        result = x_sparse * 2
        # Convert sparse tensor back to dense
        return tf.sparse.to_dense(result)

# Create a random normal tensor
x = tf.random.normal((10, 10))

model = TestModel()
print("Eager output:", model(x))
# tf.function is the TensorFlow equivalent to torch.compile for graph optimization
print("Compiled output:", tf.function(model)(x))
```