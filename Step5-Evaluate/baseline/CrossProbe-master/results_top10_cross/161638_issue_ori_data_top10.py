```python
import tensorflow as tf
import numpy as np

# Conversion: torch.tensor -> tf.constant for data
# Conversion: -torch.inf -> -np.inf
x = tf.constant([[1., 2., -np.inf], [2., 1., -np.inf]], dtype=tf.float32)

# Conversion: requires_grad=True -> tf.Variable
t = tf.Variable(1.0, dtype=tf.float32)

# Conversion: torch.logsumexp(..., dim=0) -> tf.reduce_logsumexp(..., axis=0)
# Note: TensorFlow uses GradientTape for automatic differentiation
with tf.GradientTape() as tape:
    y = tf.reduce_logsumexp(x / t, axis=0)

print(f"{y=}")

# Conversion: y[y.isfinite()] -> tf.boolean_mask(y, tf.math.is_finite(y))
# Conversion: .mean() -> tf.reduce_mean(...)
# Note: z calculation must be inside the tape context to track gradients
with tf.GradientTape() as tape:
    y = tf.reduce_logsumexp(x / t, axis=0)
    z = tf.reduce_mean(tf.boolean_mask(y, tf.math.is_finite(y))) # All non-finite values excluded in the calculation of z

# Conversion: z.backward() -> tape.gradient(z, t)
# Conversion: t.grad -> result of tape.gradient (TF Variables do not store .grad attribute)
t_grad = tape.gradient(z, t)
print(f"t_grad={t_grad}")

# Conversion: t.grad = None -> Not needed in TF as gradients are not persistent on the variable
# We start a new tape for the next operation
with tf.GradientTape() as tape:
    # Conversion: x[:, :2] slicing works identically
    y2 = tf.reduce_logsumexp(x[:, :2] / t, axis=0) # If non-finite values are excluded before logsumexp, all works fine

print(f"{y2=}")

with tf.GradientTape() as tape:
    y2 = tf.reduce_logsumexp(x[:, :2] / t, axis=0)
    z2 = tf.reduce_mean(tf.boolean_mask(y2, tf.math.is_finite(y2)))

# Conversion: z2.backward() -> tape.gradient(z2, t)
t_grad = tape.gradient(z2, t)
print(f"t_grad={t_grad}")
```