```python
import tensorflow as tf
import numpy as np

# Conversion: torch.tensor -> tf.constant
# Conversion: -torch.inf -> -np.inf
x = tf.constant([[1., 2., -np.inf], [2., 1., -np.inf]])

# Conversion: torch.tensor(..., requires_grad=True) -> tf.Variable
t = tf.Variable(1.0, dtype=tf.float32)

# Conversion: torch.logsumexp(..., dim=0) -> tf.math.reduce_logsumexp(..., axis=0)
# Note: Gradients are tracked within a GradientTape context
with tf.GradientTape() as tape:
    y = tf.math.reduce_logsumexp(x / t, axis=0)
    print(f"{y=}")

    # Conversion: y[y.isfinite()].mean() -> tf.reduce_mean(tf.boolean_mask(y, tf.math.is_finite(y)))
    # All non-finite values excluded in the calculation of z
    z = tf.reduce_mean(tf.boolean_mask(y, tf.math.is_finite(y)))

# Conversion: z.backward() -> tape.gradient(z, t)
t_grad = tape.gradient(z, t)
print(f"t.grad={t_grad}")

# Conversion: t.grad = None -> In TensorFlow, gradients are not stored on the variable.
# We simply start a new GradientTape context for the next computation.
with tf.GradientTape() as tape2:
    # Conversion: x[:, :2] -> x[:, :2]
    # If non-finite values are excluded before logsumexp, all works fine
    y2 = tf.math.reduce_logsumexp(x[:, :2] / t, axis=0)
    print(f"{y2=}")

    z2 = tf.reduce_mean(tf.boolean_mask(y2, tf.math.is_finite(y2)))

t_grad2 = tape2.gradient(z2, t)
print(f"t.grad={t_grad2}")
```