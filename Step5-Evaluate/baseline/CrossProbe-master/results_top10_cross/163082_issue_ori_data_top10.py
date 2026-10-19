```python
import tensorflow as tf

# Conversion: torch.manual_seed -> tf.random.set_seed
tf.random.set_seed(1337)

# Conversion: @torch.compile -> @tf.function
@tf.function
def vec_norm(e_dist):
    # Conversion: torch.nn.functional.normalize (defaults to L2 norm over dim 1)
    # tf.linalg.normalize returns a tuple (normalized_tensor, norm)
    normed, _ = tf.linalg.normalize(e_dist, ord=2, axis=1)
    return normed

def vec_norm_without_compile(e_dist):
    normed, _ = tf.linalg.normalize(e_dist, ord=2, axis=1)
    return normed

# Conversion: device='cuda' -> TensorFlow uses GPU automatically if available
# Conversion: torch.tensor -> tf.constant
c = tf.constant([[3.799999 ,0.0, 0.0]], dtype=tf.float32)

# Conversion: .item() -> .numpy() for scalar extraction
print("Input vector", c[0].numpy().tolist())

xyz = vec_norm(c)
# Conversion: torch.acos -> tf.math.acos
print("Normalized vector (compile):", xyz[0].numpy().tolist(), "tf.math.acos of component 0", tf.math.acos(xyz[0, 0]).numpy())

xyz = vec_norm_without_compile(c)
print("Normalized vector (without compile):", xyz[0].numpy().tolist(), "tf.math.acos of component 0", tf.math.acos(xyz[0, 0]).numpy())
```