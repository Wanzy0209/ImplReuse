import sys
import torch
import numpy as np

try:
    import tensorflow as tf
except ImportError as e:
    # Handle environment dependency issues (e.g., missing GLIBCXX)
    print(f"Skipping test: TensorFlow environment is not configured correctly. Error: {e}")
    sys.exit(0)

# Set seed for reproducibility, mirroring the original issue
tf.random.set_seed(1337)

# Define the compiled version (analogous to @torch.compile)
@tf.function
def kl_div_compiled(y_true, y_pred):
    return tf.keras.losses.kl_divergence(y_true, y_pred)

# Define the eager version (analogous to vec_norm_without_compile)
def kl_div_eager(y_true, y_pred):
    return tf.keras.losses.kl_divergence(y_true, y_pred)

# Define inputs. The original issue used [3.799999, 0.0, 0.0] with float32.
# For KL divergence, we use probability-like values that might stress precision,
# specifically values close to 0 or 1, and using float32 to match the original bug's dtype.
y_true = tf.constant([[0.5, 0.5, 0.0]], dtype=tf.float32)
y_pred = tf.constant([[0.5000001, 0.4999999, 0.0]], dtype=tf.float32)

print("Input y_true:", y_true.numpy()[0])
print("Input y_pred:", y_pred.numpy()[0])

# Run compiled
loss_compiled = kl_div_compiled(y_true, y_pred)
print("Loss (compiled):", loss_compiled.numpy()[0])

# Run eager
loss_eager = kl_div_eager(y_true, y_pred)
print("Loss (eager):", loss_eager.numpy()[0])

# Check for numerical stability (no NaNs or Infs)
# The original bug resulted in a norm > 1, which is a form of numerical instability.
assert not tf.reduce_any(tf.math.is_nan(loss_compiled)).numpy(), "Compiled loss produced NaN"
assert not tf.reduce_any(tf.math.is_inf(loss_compiled)).numpy(), "Compiled loss produced Inf"

# Check consistency between compiled and eager execution
# The original bug was a discrepancy where compiled output differed from non-compiled.
# We assert that the difference is within a reasonable floating-point epsilon.
diff = tf.abs(loss_compiled - loss_eager)
assert tf.reduce_all(diff < 1e-6).numpy(), f"Compiled and eager results differ too much: {diff.numpy()}"