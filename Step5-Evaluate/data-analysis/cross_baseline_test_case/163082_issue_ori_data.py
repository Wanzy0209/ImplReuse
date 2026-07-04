```python
import tensorflow as tf

# Conversion: torch.manual_seed -> tf.random.set_seed
# Note: Context suggested 'from_seed', but 'set_seed' is the correct global seed equivalent.
tf.random.set_seed(1337)

# Conversion: torch.compile -> tf.function
# Note: Context suggested 'get_key_counter_alg', but 'tf.function' is the correct compilation decorator.
@tf.function
def vec_norm(e_dist):
    # Conversion: torch.nn.functional.normalize -> tf.linalg.normalize
    # Note: Context suggested 'less', but 'normalize' is required for L2 normalization.
    # PyTorch default dim is the first dimension with size > 1 (dim=1 for 2D tensor).
    return tf.linalg.normalize(e_dist, axis=1)[0]

def vec_norm_without_compile(e_dist):
    return tf.linalg.normalize(e_dist, axis=1)[0]

# Conversion: device='cuda' -> device='/GPU:0'
# Note: TensorFlow uses string device identifiers.
device = '/GPU:0'
with tf.device(device):
    # Conversion: torch.tensor -> tf.constant
    # Note: Context suggested 'flush', but 'tf.constant' creates the tensor.
    c = tf.constant([[3.799999, 0.0, 0.0]], dtype=tf.float32)

# Conversion: .item() iteration -> .numpy().tolist()
print("Input vector", c[0].numpy().tolist())

xyz = vec_norm(c)
# Conversion: torch.acos -> tf.math.acos
print("Normalized vector (compile):", xyz[0].numpy().tolist(), "tf.math.acos of component 0", tf.math.acos(xyz[0, 0]).numpy())

xyz = vec_norm_without_compile(c)
print("Normalized vector (without compile):", xyz[0].numpy().tolist(), "tf.math.acos of component 0", tf.math.acos(xyz[0, 0]).numpy())
```