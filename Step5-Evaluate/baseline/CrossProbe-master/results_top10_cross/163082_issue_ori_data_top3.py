```python
import tensorflow as tf

# Conversion: torch.manual_seed -> tf.random.set_seed
tf.random.set_seed(1337)

# Conversion: @torch.compile -> @tf.function
@tf.function
def vec_norm(e_dist):
    # Conversion: torch.nn.functional.normalize (default p=2, dim=1) -> tf.math.l2_normalize
    return tf.math.l2_normalize(e_dist, axis=1)

def vec_norm_without_compile(e_dist):
    return tf.math.l2_normalize(e_dist, axis=1)

# Conversion: device='cuda' -> tf.device('/GPU:0')
device = '/GPU:0'
with tf.device(device):
    # Conversion: torch.tensor -> tf.constant
    c = tf.constant([[3.799999, 0.0, 0.0]], dtype=tf.float32)

# Conversion: .item() -> .numpy().item() for scalar tensors
print("Input vector", [x.numpy().item() for x in c[0]])
xyz = vec_norm(c)
print("Normalized vector (compile):", [x.numpy().item() for x in xyz[0]], "tf.math.acos of component 0", tf.math.acos(xyz[0, 0]).numpy().item())
xyz = vec_norm_without_compile(c)
print("Normalized vector (without compile):", [x.numpy().item() for x in xyz[0]], "tf.math.acos of component 0", tf.math.acos(xyz[0, 0]).numpy().item())
```