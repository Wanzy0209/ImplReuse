```python
import tensorflow as tf

# Conversion: torch.tensor -> tf.constant for data, tf.Variable for trainable parameters
# Conversion: -torch.inf -> -float('inf')
x = tf.constant([[1., 2., -float('inf')], [2., 1., -float('inf')]], dtype=tf.float32)
# Conversion: requires_grad=True -> tf.Variable (trainable=True by default)
t = tf.Variable(1.0, dtype=tf.float32)

# Conversion: torch.logsumexp(..., dim=0) -> tf.math.reduce_logsumexp(..., axis=0)
# Note: TensorFlow uses GradientTape for automatic differentiation
with tf.GradientTape() as tape:
    y = tf.math.reduce_logsumexp(x / t, axis=0)
    print(f"y={y}")

    # Conversion: y[y.isfinite()] -> tf.boolean_mask(y, tf.math.is_finite(y))
    # Conversion: .mean() -> tf.reduce_mean
    z = tf.reduce_mean(tf.boolean_mask(y, tf.math.is_finite(y))) # All non-finite values excluded in the calculation of z

# Conversion: z.backward() -> tape.gradient(z, t)
# Conversion: t.grad -> t_grad (TF Variables don't store gradients as attributes)
t_grad = tape.gradient(z, t)
print(f"t.grad={t_grad}")

# Conversion: t.grad = None -> Reset local gradient variable
# (TF gradients are not accumulated on the Variable object itself)
t_grad = None

with tf.GradientTape() as tape:
    # Conversion: x[:, :2] -> x[:, :2]
    y2 = tf.math.reduce_logsumexp(x[:, :2] / t, axis=0) # If non-finite values are excluded before logsumexp, all works fine
    print(f"y2={y2}")

    z2 = tf.reduce_mean(tf.boolean_mask(y2, tf.math.is_finite(y2)))

t_grad = tape.gradient(z2, t)
print(f"t.grad={t_grad}")
```