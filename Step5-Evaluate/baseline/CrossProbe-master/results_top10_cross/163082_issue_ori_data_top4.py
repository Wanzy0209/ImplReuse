```python
import tensorflow as tf

# Conversion: torch.manual_seed -> tf.random.set_seed
tf.random.set_seed(1337)

# Conversion: @torch.compile -> @tf.function
@tf.function
def vec_norm(e_dist):
    # Conversion: torch.nn.functional.normalize -> tf.linalg.l2_normalize
    # PyTorch default is L2 norm over dimension 1
    return tf.linalg.l2_normalize(e_dist, axis=1)

def vec_norm_without_compile(e_dist):
    return tf.linalg.l2_normalize(e_dist, axis=1)

# Conversion: device='cuda' -> Check for GPU availability
device = '/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'

# Conversion: torch.tensor -> tf.constant
# Note: TF uses with tf.device() context for placement
with tf.device(device):
    c = tf.constant([[3.799999 ,0.0, 0.0]], dtype=tf.float32)

# Conversion: x.item() -> x.numpy()
print("Input vector", [x for x in c[0].numpy()])

xyz = vec_norm(c)
# Conversion: torch.acos -> tf.math.acos
print("Normalized vector (compile):", [x for x in xyz[0].numpy()], 
      "tf.math.acos of component 0", tf.math.acos(xyz[0, 0]).numpy())

xyz = vec_norm_without_compile(c)
print("Normalized vector (without compile):", [x for x in xyz[0].numpy()], 
      "tf.math.acos of component 0", tf.math.acos(xyz[0, 0]).numpy())
```