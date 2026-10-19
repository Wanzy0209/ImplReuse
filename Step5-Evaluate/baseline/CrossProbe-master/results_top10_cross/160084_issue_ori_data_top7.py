```python
import tensorflow as tf

class RegressionModel(tf.Module):  # torch.nn.Module -> tf.Module
    def __init__(self, a=0, b=0):
        super().__init__()
        # torch.nn.Parameter -> tf.Variable
        # torch.tensor(a).float() -> tf.cast(a, tf.float32)
        self.a = tf.Variable(tf.cast(a, tf.float32))
        self.b = tf.Variable(tf.cast(b, tf.float32))
        self.first_batch = True

    def forward(self, x=None):
        if self.first_batch:
            # Note: In tf.function, print() executes during tracing.
            # tf.print() is recommended for runtime execution, but print() preserves the original logic here.
            print(f"Model dtype: {self.a.dtype}, {self.b.dtype}. Input dtype: {x.dtype}")
            self.first_batch = False
        return x * self.a + self.b

model = RegressionModel()
# TensorFlow handles device placement automatically (GPU if available)
torch_device = "cuda"
# torch.compile -> tf.function
model.forward = tf.function(model.forward)
# torch.randn -> tf.random.normal
# .to(torch_device) is implicit in TF 2.x if GPU is available
inputs = tf.random.normal((4, 10))
_ = model.forward(inputs)
```