```python
import tensorflow as tf

class RegressionModel(tf.Module):  # Conversion: torch.nn.Module -> tf.Module
    def __init__(self, a=0, b=0):
        super().__init__()
        # Conversion: torch.nn.Parameter -> tf.Variable (trainable=True)
        # Conversion: torch.tensor(a).float() -> tf.cast(a, tf.float32)
        self.a = tf.Variable(tf.cast(a, tf.float32), trainable=True)
        self.b = tf.Variable(tf.cast(b, tf.float32), trainable=True)
        self.first_batch = True

    def forward(self, x=None):
        if self.first_batch:
            print(f"Model dtype: {self.a.dtype}, {self.b.dtype}. Input dtype: {x.dtype}")
            self.first_batch = False
        return x * self.a + self.b

model = RegressionModel()
# Conversion: "cuda" -> "/GPU:0"
tf_device = "/GPU:0"
# Conversion: torch.compile -> tf.function
model.forward = tf.function(model.forward)

# Conversion: torch.randn -> tf.random.normal
# Conversion: .to(torch_device) -> with tf.device(...)
with tf.device(tf_device):
    inputs = tf.random.normal((4, 10))

_ = model(inputs)
```