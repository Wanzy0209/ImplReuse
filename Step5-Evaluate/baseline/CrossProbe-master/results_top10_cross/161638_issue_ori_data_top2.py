```python
import tensorflow as tf
import numpy as np

# Conversion: torch.tensor -> tf.constant
# Conversion: -torch.inf -> -np.inf
x = tf.constant([[1., 2., -np.inf], [2., 1., -np.inf]], dtype=tf.float32)

# Conversion: torch.tensor(..., requires_grad=True) -> tf.Variable
t = tf.Variable(1.0, dtype=tf.float32)

# Conversion: Gradients in TensorFlow are tracked using tf.GradientTape
with tf.GradientTape() as tape:
    # Conversion: torch.logsumexp(..., dim=0) -> tf.math.reduce_logsumexp(..., axis=0)
    y = tf.math.reduce_logsumexp(x / t, axis=0)
    print(f"{y=}")

    # Conversion: y[y.isfinite()] -> tf.boolean_mask(y, tf.math.is_finite(y))
    z = tf.reduce_mean(tf.boolean_mask(y, tf.math.is_finite(y))) # All non-finite values excluded in the calculation of z

# Conversion: z.backward() -> tape.gradient(target, source)
t_grad = tape.gradient(z, t)
print(f"t.grad={t_grad}")

# Conversion: t.grad = None -> Not strictly required in TF as gradients are calculated per tape context
with tf.GradientTape() as tape:
    # Conversion: x[:, :2] slicing is identical
    y2 = tf.math.reduce_logsumexp(x[:, :2] / t, axis=0) # If non-finite values are excluded before logsumexp, all works fine
    print(f"{y2=}")
    z2 = tf.reduce_mean(tf.boolean_mask(y2, tf.math.is_finite(y2)))

t_grad = tape.gradient(z2, t)
print(f"t.grad={t_grad}")
```