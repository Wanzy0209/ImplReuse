import torch
import tensorflow as tf

def inner(x):
    return x + 1

@tf.function
def fn(x):
    # Leverage tf.control_dependencies to manage execution context,
    # reflecting the need for explicit tracking/control mentioned in the issue.
    with tf.control_dependencies([]):
        x = inner(x)
    return inner(x)

# Original: fn(torch.ones(3))
# Replacement: torch.ones -> torch.norm
# Translated: fn(tf.norm(tf.ones(3)))
fn(tf.norm(tf.ones(3)))