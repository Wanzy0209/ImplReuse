```python
import tensorflow as tf

class Isin(tf.Module):
    def __init__(self, device):
        super().__init__()
        # Conversion: torch.manual_seed(777) -> tf.random.set_seed(777)
        tf.random.set_seed(777)
        
        # Conversion: torch.device handling
        # TF uses device context managers. We map 'cuda' to the first GPU if available.
        target_device = '/GPU:0' if device == 'cuda' and tf.config.list_physical_devices('GPU') else '/CPU:0'
        
        with tf.device(target_device):
            # Conversion: torch.randint(low, high, size, dtype) -> tf.random.uniform(shape, minval, maxval, dtype)
            # Note: torch.randint is exclusive of high, tf.random.uniform is exclusive of maxval.
            self.x = tf.random.uniform(shape=(1,), minval=-50, maxval=50, dtype=tf.int64)
            self.y = tf.random.uniform(shape=(), minval=-50, maxval=50, dtype=tf.int64)

    def forward(self):
        # Conversion: print -> tf.print
        # tf.print is used here to ensure values are printed in both eager and graph (compiled) modes.
        tf.print(self.x)
        tf.print(self.y)
        
        # Conversion: torch.isin(self.x, self.y, assume_unique=False, invert=False)
        # Since self.y is a scalar in this specific case, tf.equal is sufficient.
        # For general tensors, one would use: tf.reduce_any(tf.equal(tf.expand_dims(self.x, -1), self.y), axis=-1)
        out = tf.equal(self.x, self.y)
        
        return {'out': out}

# Determine device (mimicking 'cuda' check)
device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
model = Isin(device)

print("Eager:", model.forward())

# Conversion: torch.compile(model, backend='inductor') -> tf.function
# In TensorFlow, we decorate the function or wrap it with tf.function to enable graph execution/compilation.
compiled_forward = tf.function(model.forward)
print("Compiled:", compiled_forward())
```