```python
import tensorflow as tf

# Conversion: torch.tensor -> tf.constant (explicitly set dtype to float32 to match PyTorch behavior)
x = tf.constant([[1., 2., -float('inf')], [2., 1., -float('inf')]], dtype=tf.float32)
# Conversion: torch.tensor(..., requires_grad=True) -> tf.Variable
t = tf.Variable(1.0, dtype=tf.float32)

with tf.GradientTape() as tape:
    # Conversion: torch.logsumexp(..., dim=0) -> tf.math.reduce_logsumexp(..., axis=0)
    y = tf.math.reduce_logsumexp(x / t, axis=0)
    print(f"{y=}")

    # Conversion: y[y.isfinite()] -> tf.boolean_mask(y, tf.math.is_finite(y))
    # All non-finite values excluded in the calculation of z
    z = tf.reduce_mean(tf.boolean_mask(y, tf.math.is_finite(y)))

# Conversion: z.backward() -> tape.gradient(target, source)
t_grad = tape.gradient(z, t)
print(f"{t_grad=}")

# t.grad = None # Not strictly required in TF as we use a new tape context
with tf.GradientTape() as tape:
    # Conversion: x[:, :2] slicing works the same in TF
    # If non-finite values are excluded before logsumexp, all works fine
    y2 = tf.math.reduce_logsumexp(x[:, :2] / t, axis=0)
    print(f"{y2=}")

    z2 = tf.reduce_mean(tf.boolean_mask(y2, tf.math.is_finite(y2)))

t_grad = tape.gradient(z2, t)
print(f"{t_grad=}")
```