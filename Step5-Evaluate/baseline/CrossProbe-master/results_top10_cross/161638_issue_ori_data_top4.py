```python
import tensorflow as tf
import numpy as np

# Conversion: torch.tensor -> tf.constant
# Conversion: -torch.inf -> -np.inf
x = tf.constant([[1., 2., -np.inf], [2., 1., -np.inf]], dtype=tf.float32)

# Conversion: torch.tensor(..., requires_grad=True) -> tf.Variable
t = tf.Variable(1.0, dtype=tf.float32)

# Conversion: torch.logsumexp(..., dim=0) -> tf.math.reduce_logsumexp(..., axis=0)
# Note: In TensorFlow, gradient computation requires a tf.GradientTape context.
with tf.GradientTape() as tape:
    # x / t is element-wise division
    y = tf.math.reduce_logsumexp(x / t, axis=0)
    print(f"{y=}")

    # Conversion: y[y.isfinite()] -> tf.boolean_mask(y, tf.math.is_finite(y))
    # Conversion: .mean() -> tf.reduce_mean(...)
    # All non-finite values excluded in the calculation of z
    z = tf.reduce_mean(tf.boolean_mask(y, tf.math.is_finite(y)))

# Conversion: z.backward() -> tape.gradient(z, t)
# Conversion: t.grad -> tape.gradient(...)
print(f"t.grad={tape.gradient(z, t)}")

# Conversion: t.grad = None
# Note: TensorFlow does not accumulate gradients on Variables, so clearing is not required.
# t.grad = None

# Conversion: x[:, :2] -> x[:, :2] (slicing is similar)
# If non-finite values are excluded before logsumexp, all works fine
with tf.GradientTape() as tape:
    y2 = tf.math.reduce_logsumexp(x[:, :2] / t, axis=0)
    print(f"{y2=}")

    z2 = tf.reduce_mean(tf.boolean_mask(y2, tf.math.is_finite(y2)))

print(f"t.grad={tape.gradient(z2, t)}")
```