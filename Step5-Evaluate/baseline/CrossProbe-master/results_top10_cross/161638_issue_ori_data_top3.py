```python
import tensorflow as tf
import numpy as np

# Conversion: torch.tensor -> tf.constant
# Conversion: -torch.inf -> -np.inf
x = tf.constant([[1., 2., -np.inf], [2., 1., -np.inf]], dtype=tf.float32)

# Conversion: torch.tensor(..., requires_grad=True) -> tf.Variable
t = tf.Variable(1.0, dtype=tf.float32)

# Conversion: torch.logsumexp(..., dim=0) -> tf.reduce_logsumexp(..., axis=0)
# Note: In TensorFlow, gradients are tracked using tf.GradientTape.
# The operations involving the variable 't' must be inside the tape context.
with tf.GradientTape() as tape:
    y = tf.reduce_logsumexp(x / t, axis=0)
    print(f"{y=}")

    # Conversion: y[y.isfinite()].mean() -> tf.reduce_mean(tf.boolean_mask(y, tf.math.is_finite(y)))
    # All non-finite values excluded in the calculation of z
    z = tf.reduce_mean(tf.boolean_mask(y, tf.math.is_finite(y)))

# Conversion: z.backward() -> tape.gradient(z, t)
# Note: In TF, gradients are returned rather than stored on the tensor
t_grad = tape.gradient(z, t)
print(f"t_grad={t_grad}")

# Conversion: t.grad = None
# In TF, gradients are not accumulated on the variable, so we just reset the local variable
t_grad = None

# Conversion: torch.logsumexp(..., dim=0) -> tf.reduce_logsumexp(..., axis=0)
# If non-finite values are excluded before logsumexp, all works fine
with tf.GradientTape() as tape:
    y2 = tf.reduce_logsumexp(x[:, :2] / t, axis=0)
    print(f"{y2=}")
    
    z2 = tf.reduce_mean(tf.boolean_mask(y2, tf.math.is_finite(y2)))

t_grad = tape.gradient(z2, t)
print(f"t_grad={t_grad}")
```