import torch
import tensorflow as tf
import functools

# Adapted from torch.utils.checkpoint.CheckpointPolicy
# The target API is tf.compat.v1.distributions.ReparameterizationType
from tf.compat.v1.distributions import ReparameterizationType

# Mimic the CustomPolicy class from the PyTorch bug report
class CustomReparamPolicy:
    def __init__(self):
        super().__init__()

    def __call__(self):
        # Return a specific reparameterization type, similar to returning CheckpointPolicy.MUST_SAVE
        return ReparameterizationType.FULLY_REPARAMETERIZED

# Mimic create_selective_checkpoint_contexts
def get_reparam_context(policy_fn):
    return policy_fn()

# Create the partial function, which is the core of the bug report
# In PyTorch: context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())
# Here: reparam_fn = functools.partial(get_reparam_context, CustomReparamPolicy())
reparam_fn = functools.partial(get_reparam_context, CustomReparamPolicy())

# Mimic the function f(x, y) from PyTorch
def f(x):
    return tf.sigmoid(tf.matmul(x, x)) * x

# Mimic @torch.compile with @tf.function
@tf.function
def g(x):
    # Retrieve the reparameterization type using the partial function
    # This tests if tf.function can trace the functools.partial correctly
    r_type = reparam_fn()
    
    # Create a distribution that respects the reparameterization type
    # Normal is fully reparameterized, so we check against the policy
    dist = tf.compat.v1.distributions.Normal(loc=x, scale=1.0)
    
    # Verify the distribution type matches the policy type
    # This acts as the assertion logic inside the compiled graph
    assert dist.reparameterization_type == r_type
    
    # Perform a forward pass (mimicking the checkpointed function)
    return f(x)

# Setup inputs
a = tf.ones((4, 4), dtype=tf.float32)

# Run the compiled function and compute gradients
# This mimics g(a, b).sum().backward()
with tf.GradientTape() as tape:
    result = g(a)
    loss = tf.reduce_sum(result)

grads = tape.gradient(loss, a)

# Verify gradients are computed (reparameterization works)
assert grads is not None
print("Test passed: functools.partial with ReparameterizationType handled correctly in tf.function.")