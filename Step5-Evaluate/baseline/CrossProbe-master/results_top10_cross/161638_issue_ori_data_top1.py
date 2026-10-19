```python
import tensorflow as tf
import numpy as np

# Conversion: torch.tensor -> tf.constant
# Conversion: -torch.inf -> -np.inf
x = tf.constant([[1., 2., -np.inf], [2., 1., -np.inf]])

# Conversion: torch.tensor(..., requires_grad=True) -> tf.Variable
t = tf.Variable(1.0, dtype=tf.float32)

# Conversion: PyTorch autograd requires a GradientTape context
with tf.GradientTape() as tape:
    # Conversion: torch.logsumexp(..., dim=0) -> tf.math.reduce_logsumexp(..., axis=0)
    y = tf.math.reduce_logsumexp(x / t, axis=0)
    print(f"{y=}")

    # Conversion: y[y.isfinite()] -> tf.boolean_mask(y, tf.math.is_finite(y))
    # All non-finite values excluded in the calculation of z
    z = tf.reduce_mean(tf.boolean_mask(y, tf.math.is_finite(y)))
    
    # Conversion: z.backward() -> tape.gradient(z, t)
    grads = tape.gradient(z, t)

# Conversion: t.grad -> grads (TF returns gradients, they are not stored on the variable)
print(f"t.grad={grads}")

# Conversion: t.grad = None -> Not strictly needed in TF as a new GradientTape context 
# resets the computation graph, but we can clear the local variable if desired.
grads = None

with tf.GradientTape() as tape:
    # Conversion: x[:, :2] slicing works the same in TF
    # If non-finite values are excluded before logsumexp, all works fine
    y2 = tf.math.reduce_logsumexp(x[:, :2] / t, axis=0)
    print(f"{y2=}")
    
    z2 = tf.reduce_mean(tf.boolean_mask(y2, tf.math.is_finite(y2)))
    grads2 = tape.gradient(z2, t)

print(f"t.grad={grads2}")
```