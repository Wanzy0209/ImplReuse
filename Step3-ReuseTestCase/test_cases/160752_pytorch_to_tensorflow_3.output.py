import torch
import tensorflow as tf

# API Call: Enable eager execution
# This is the TensorFlow API identified as similar to the context of the PyTorch bug.
tf.compat.v1.enable_eager_execution()

# Constants from the original bug report
MAX = 3
BATCH = 37

# Define the function logic
def func(x, idxs):
    # PyTorch: x.square() * torch.nn.functional.one_hot(idxs, MAX)
    # TensorFlow: tf.square(x) * tf.one_hot(idxs, MAX)
    return tf.square(x) * tf.one_hot(idxs, MAX)

# Prepare data
# PyTorch: torch.randint(MAX, (BATCH,), dtype=torch.int64)
idxs = tf.random.uniform((BATCH,), maxval=MAX, dtype=tf.int64)
# PyTorch: torch.rand((BATCH, MAX), dtype=torch.float64)
x = tf.random.uniform((BATCH, MAX), dtype=tf.float64)

# Define Jacobian calculation
# PyTorch: torch.func.jacfwd(func, argnums=(0,))
# TensorFlow: tf.jacobian
def jacfunc(x_val):
    # We need a wrapper because tf.jacobian expects f(x)
    return func(x_val, idxs)

# Execute
# In the original PyTorch bug, the uncompiled version works, but the compiled one fails.
# Here, we verify the behavior under eager execution enabled by the target API.
try:
    # Calculate Jacobian with respect to x
    jacobian_result = tf.jacobian(jacfunc, x)
    
    # Basic assertion to ensure execution happened and shapes are correct
    # Output of func is (BATCH, MAX), Input x is (BATCH, MAX)
    # Jacobian shape should be (BATCH, MAX, BATCH, MAX)
    expected_shape = (BATCH, MAX, BATCH, MAX)
    assert jacobian_result.shape == expected_shape, \
        f"Shape mismatch: expected {expected_shape}, got {jacobian_result.shape}"
    
    print("Test passed: Jacobian computed successfully with eager execution enabled.")

except Exception as e:
    print(f"Test failed with error: {e}")
    raise