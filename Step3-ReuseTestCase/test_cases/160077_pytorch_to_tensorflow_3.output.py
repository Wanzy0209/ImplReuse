import torch
import tensorflow as tf

# Define the function using the similar API (tf.nn.dropout)
# This mirrors the structure of the original PyTorch function 'f'
def f(x):
    # tf.nn.dropout applies dropout to the input tensor.
    # rate=0.5 means zero out 50% of the elements.
    # seed is set for reproducibility.
    return tf.nn.dropout(x, rate=0.5, seed=42)

# Determine a valid device to ensure the test is runnable on various environments.
# The original bug report specifically targeted "cuda" (GPU).
# We attempt to use GPU if available, otherwise fallback to CPU to maintain runnability.
device_name = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

# The core bug reproduction logic involves executing the function inside a device context
# and then attempting to compile/trace it within that same context.
with tf.device(device_name):
    # Create input tensor
    # Equivalent to: xs = torch.randn(2, 2, device="cuda")
    x = tf.random.normal((2, 2))

    # 1. Eager execution
    # Equivalent to: f(xs)
    print("Testing eager execution...")
    eager_result = f(x)
    # Basic assertion to verify the API ran successfully
    assert eager_result.shape == x.shape, "Eager execution failed: shape mismatch"

    # 2. Compiled execution
    # Equivalent to: torch.compile(f, backend=backend)(xs)
    # In TensorFlow, tf.function compiles the Python function into a static graph.
    print("Testing compiled execution (tf.function)...")
    compiled_f = tf.function(f)
    compiled_result = compiled_f(x)
    # Basic assertion to verify the compiled API ran successfully
    assert compiled_result.shape == x.shape, "Compiled execution failed: shape mismatch"

print("Test passed: tf.nn.dropout works inside device context under both eager and compiled modes.")