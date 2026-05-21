import torch
import tensorflow as tf

# Enable eager execution for functions, analogous to backend="eager" in torch.compile
tf.config.run_functions_eagerly(True)

def inner(x):
    # The PyTorch issue requests logging variable types/states during tracing.
    # Here we use the similar API to inspect the execution state (eager vs graph).
    # This provides the "debug information" regarding the execution context.
    return x + 1, tf.executing_eagerly()

@tf.function
def fn(x):
    # Replicate the structure of the original bug report:
    # calling the inner function twice to observe state changes.
    x, is_eager_1 = inner(x)
    x, is_eager_2 = inner(x)
    return x, is_eager_1, is_eager_2

# Execute the test
# Outside the function, execution is eager by default
assert tf.executing_eagerly() is True

# Call the compiled function
result, eager_state_1, eager_state_2 = fn(tf.ones(3))

# Verify the execution state inside the function.
# Since we enabled run_functions_eagerly, we expect True, 
# mirroring the context of the PyTorch issue where the backend is eager.
assert eager_state_1 is True, "Expected eager execution in first inner call"
assert eager_state_2 is True, "Expected eager execution in second inner call"