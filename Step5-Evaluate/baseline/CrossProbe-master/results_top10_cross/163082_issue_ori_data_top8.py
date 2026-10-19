```python
import tensorflow as tf

# Conversion: torch.manual_seed -> tf.random.set_seed
tf.random.set_seed(1337)

# Conversion: @torch.compile -> @tf.function
@tf.function
def vec_norm(e_dist):
    # Conversion: torch.nn.functional.normalize (default dim=1, p=2) -> tf.linalg.l2_normalize (axis=1)
    return tf.linalg.l2_normalize(e_dist, axis=1)

def vec_norm_without_compile(e_dist):
    return tf.linalg.l2_normalize(e_dist, axis=1)

# Conversion: device='cuda' -> with tf.device('/GPU:0')
device = '/GPU:0'
with tf.device(device):
    # Conversion: torch.tensor -> tf.constant
    c = tf.constant([[3.799999 ,0.0, 0.0]], dtype=tf.float32)

# Conversion: .item() -> float() conversion for printing
print("Input vector", [float(x) for x in c[0] ])
xyz = vec_norm(c)
# Conversion: torch.acos -> tf.math.acos
print("Normalized vector (compile):", [float(x) for x in xyz[0] ], "torch.acos of component 0", float(tf.math.acos(xyz[0, 0])))
xyz = vec_norm_without_compile(c)
print("Normalized vector (without compile):", [float(x) for x in xyz[0] ], "torch.acos of component 0", float(tf.math.acos(xyz[0, 0])))
```