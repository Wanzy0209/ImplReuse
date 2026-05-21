import torch
import tensorflow as tf
import functools

# Define a simple body function that takes an extra argument
# This mimics the structure of create_selective_checkpoint_contexts taking a policy
def loop_body(x, multiplier):
    return x * multiplier

# Define a condition for the loop
def loop_cond(x):
    return x < 100.0

# Create a partial function fixing the 'multiplier' argument
# This mimics: context_fn1 = functools.partial(create_selective_checkpoint_contexts, CustomPolicy())
partial_body = functools.partial(loop_body, multiplier=2.0)

# Wrap in tf.function to mimic @torch.compile
# This tests if the TensorFlow tracer handles functools.partial callables correctly
@tf.function
def run_loop(initial_val):
    return tf.compat.v1.while_loop(
        loop_cond,
        partial_body,
        [initial_val]
    )

# Test execution (Forward pass)
# Initial value 1.0: 1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> 128 (Stop, 128 >= 100)
# Expected result: 64.0
var = tf.constant(1.0)
result = run_loop(var)
print("Forward pass result:", result.numpy())
assert result.numpy() == 64.0, f"Expected 64.0, got {result.numpy()}"

# Test gradients (Backward pass)
# This mimics the .backward() call in the original PyTorch test case
with tf.GradientTape() as tape:
    v_var = tf.Variable(1.0)
    res = run_loop(v_var)

grads = tape.gradient(res, v_var)
print("Gradients:", grads.numpy())
# The gradient of x * 2^n with respect to x is 2^n.
# Since the loop ran 6 times (1->64), multiplier is 2^6 = 64.
assert grads is not None, "Gradients should not be None"
assert grads.numpy() == 64.0, f"Expected gradient 64.0, got {grads.numpy()}"

print("Test passed.")