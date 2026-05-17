import tensorflow as tf
import numpy as np

# Set seeds for reproducibility
tf.random.set_seed(0)
np.random.seed(0)

# Define the Laplace distribution
# Note: The parameters from the original bug (kernel_size, stride, ceil_mode, divisor_override)
# are specific to AvgPool2d and do not apply to the Laplace distribution.
# We use standard parameters for the distribution here.
loc = 0.0
scale = 1.0
dist = tf.compat.v1.distributions.Laplace(loc=loc, scale=scale, validate_args=True)

# Create input data (matching the shape from the original bug report)
x = np.random.randn(4, 6, 7).astype(np.float32)

# Compute output using the TensorFlow API
out_tf = dist.prob(x).numpy()

# Compute expected output manually using NumPy (mimicking the "correct" CPU reference)
# PDF formula: exp(-|x - loc| / scale) / (2 * scale)
expected_out = np.exp(-np.abs(x - loc) / scale) / (2 * scale)

# Verify the results
if not np.allclose(out_tf, expected_out, atol=1e-5, rtol=1e-5):
    print("Output does not match!")
    print("Expected (NumPy):")
    print(expected_out)
    print("Actual (TensorFlow):")
    print(out_tf)
else:
    print("Test passed: Output matches expected values.")