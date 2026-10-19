```python
import tensorflow as tf

class Isin(tf.Module):
    def __init__(self, device):
        super().__init__()
        # Conversion: torch.manual_seed -> tf.random.set_seed
        tf.random.set_seed(777)
        
        # Conversion: PyTorch 'cuda' -> TensorFlow '/GPU:0' (or '/CPU:0' if unavailable)
        self.device = '/GPU:0' if device == 'cuda' and tf.config.list_physical_devices('GPU') else '/CPU:0'
        
        with tf.device(self.device):
            # Conversion: torch.randint -> tf.random.uniform
            # Note: minval is inclusive, maxval is exclusive, matching PyTorch's low/high behavior
            self.x = tf.random.uniform((1,), minval=-50, maxval=50, dtype=tf.int64)
            self.y = tf.random.uniform((), minval=-50, maxval=50, dtype=tf.int64)

    def forward(self):
        # Conversion: print -> tf.print
        # tf.print is used here to ensure values are printed in both eager and graph (compiled) modes
        tf.print(self.x)
        tf.print(self.y)
        
        # Conversion: torch.isin
        # Since self.y is a scalar (shape ()), tf.equal is the direct equivalent.
        # For general tensors, one would use: tf.reduce_any(tf.equal(tf.expand_dims(x, -1), y), axis=-1)
        out = tf.equal(self.x, self.y)
        return {'out': out}

    # Make the module callable
    def __call__(self):
        return self.forward()

# Instantiate model
model = Isin('cuda')
print("Eager:", model.forward())

# Conversion: torch.compile -> tf.function
# tf.function compiles the Python function into a static graph
compiled_model = tf.function(model)
print("Graph:", compiled_model())
```