import torch
import tensorflow as tf

def inner(x):
    return x + 1

def fn(x):
    # Adaptation: Use tf.compat.v1.name_scope to provide context for operations,
    # similar to how torch.compile provides context for the function execution.
    # We set skip_on_eager=False to ensure the scope is applied even in eager mode,
    # mirroring the behavior of the eager backend in the original test.
    with tf.compat.v1.name_scope("fn_scope", skip_on_eager=False):
        x = inner(x)
        x = inner(x)
    return x

# Execute the function
result = fn(tf.ones(3))

# Verify the behavior
# Check that the computation is correct
assert tf.reduce_all(result == tf.ones(3) + 2).numpy()

# Check that the naming context was applied (verifying the "tracking" aspect of the API)
# Note: In eager mode with skip_on_eager=False, tensors usually acquire names.
assert "fn_scope" in result.name