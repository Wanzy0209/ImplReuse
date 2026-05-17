import torch
import tensorflow as tf
import numpy as np

# Adapted test case for tf.keras.ops.vdot
# The original issue highlights gradient behavior differences in torch.min based on reduction dimensions.
# tf.keras.ops.vdot always flattens inputs (reduces over all dimensions), so we test that specific behavior.

# Setup inputs
# Using ones to mimic the PyTorch example setup
a = tf.ones([5])
b = tf.ones([5])

# Compute gradients
with tf.GradientTape(persistent=True) as tape:
    tape.watch(a)
    tape.watch(b)
    
    # vdot flattens inputs and computes dot product
    # result = sum(a_flat * b_flat)
    # With a=[1,1,1,1,1] and b=[1,1,1,1,1], result is 5.
    val = tf.keras.ops.vdot(a, b)

grad_a = tape.gradient(val, a)
grad_b = tape.gradient(val, b)

print("Gradient w.r.t a:", grad_a.numpy())
print("Gradient w.r.t b:", grad_b.numpy())

# Verify gradients
# For vdot, d(vdot)/da = b and d(vdot)/db = a
# Unlike torch.min (which distributes 1/N for ties), vdot is linear.
# The gradient is simply the other input vector.
assert np.allclose(grad_a.numpy(), b.numpy()), "Gradient w.r.t a should be b"
assert np.allclose(grad_b.numpy(), a.numpy()), "Gradient w.r.t b should be a"