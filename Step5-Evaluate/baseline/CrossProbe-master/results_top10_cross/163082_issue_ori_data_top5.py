```python
import tensorflow as tf

# Conversion: torch.manual_seed -> tf.random.set_seed
tf.random.set_seed(1337)

# Conversion: @torch.compile -> @tf.function
@tf.function
def vec_norm(e_dist):
    # Conversion: torch.nn.functional.normalize (default L2, dim=1) -> tf.nn.l2_normalize
    return tf.nn.l2_normalize(e_dist, axis=1)

def vec_norm_without_compile(e_dist):
    # Conversion: torch.nn.functional.normalize (default L2, dim=1) -> tf.nn.l2_normalize
    return tf.nn.l2_normalize(e_dist, axis=1)

# Note: TensorFlow uses GPU automatically if available. Explicit device placement is not required for basic ops.
c = tf.constant([[3.799999 ,0.0, 0.0]], dtype=tf.float32)
# Conversion: x.item() -> x.numpy() for printing tensor values
print("Input vector", [x.numpy() for x in c[0] ])

xyz = vec_norm(c)
# Conversion: torch.acos -> tf.math.acos
# Conversion: .item() -> .numpy()
print("Normalized vector (compile):", [x.numpy() for x in xyz[0] ], "tf.math.acos of component 0", tf.math.acos(xyz[0, 0]).numpy())

xyz = vec_norm_without_compile(c)
print("Normalized vector (without compile):", [x.numpy() for x in xyz[0] ], "tf.math.acos of component 0", tf.math.acos(xyz[0, 0]).numpy())
```