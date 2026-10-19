```python
import tensorflow as tf

# Conversion: torch.manual_seed -> tf.random.set_seed
tf.random.set_seed(1337)

# Conversion: @torch.compile -> @tf.function
@tf.function
def vec_norm(e_dist):
    # Conversion: torch.nn.functional.normalize -> tf.linalg.normalize
    # Note: tf.linalg.normalize returns (normalized_tensor, norm), so we take index 0.
    # PyTorch defaults to p=2, dim=1. TF defaults to axis=None, so we specify axis=1.
    return tf.linalg.normalize(e_dist, ord=2, axis=1)[0]

def vec_norm_without_compile(e_dist):
    return tf.linalg.normalize(e_dist, ord=2, axis=1)[0]

# Conversion: device='cuda' -> TF uses context managers for device placement
# We check for GPU availability to avoid errors if no GPU is present
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

with tf.device(device):
    # Conversion: torch.tensor -> tf.constant
    c = tf.constant([[3.799999, 0.0, 0.0]], dtype=tf.float32)

# Conversion: .item() -> .numpy() to extract scalar values for printing
print("Input vector", [float(x) for x in c[0].numpy()])

xyz = vec_norm(c)
# Conversion: torch.acos -> tf.math.acos
print("Normalized vector (compile):", [float(x) for x in xyz[0].numpy()], "torch.acos of component 0", float(tf.math.acos(xyz[0, 0]).numpy()))

xyz = vec_norm_without_compile(c)
print("Normalized vector (without compile):", [float(x) for x in xyz[0].numpy()], "torch.acos of component 0", float(tf.math.acos(xyz[0, 0]).numpy()))
```