```python
import tensorflow as tf

# Conversion: torch.tensor -> tf.constant
x = tf.constant([[1., 2., -float('inf')], [2., 1., -float('inf')]])

# Conversion: torch.tensor with requires_grad -> tf.Variable
t = tf.Variable(1., dtype=tf.float32)

# Conversion: torch.logsumexp -> tf.math.reduce_logsumexp
# Note: Operations must be recorded within a GradientTape context for backprop
with tf.GradientTape() as tape:
    y = tf.math.reduce_logsumexp(x / t, axis=0)
    print(f"{y=}")

    # Conversion: y[y.isfinite()] -> tf.boolean_mask(y, tf.math.is_finite(y))
    z = tf.math.reduce_mean(tf.boolean_mask(y, tf.math.is_finite(y))) # All non-finite values excluded in the calculation of z

# Conversion: z.backward() -> tape.gradient(target, source)
t_grad = tape.gradient(z, t)
print(f"{t_grad=}")

# Conversion: t.grad = None
# Note: In TensorFlow, gradients are not stored on the Variable between Tape contexts,
# so manual clearing is not necessary.
# t.grad = None

with tf.GradientTape() as tape:
    # Conversion: x[:, :2] slicing works identically
    y2 = tf.math.reduce_logsumexp(x[:, :2] / t, axis=0) # If non-finite values are excluded before logsumexp, all works fine
    print(f"{y2=}")
    z2 = tf.math.reduce_mean(tf.boolean_mask(y2, tf.math.is_finite(y2)))

t_grad = tape.gradient(z2, t)
print(f"{t_grad=}")
```