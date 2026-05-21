import torch
import tensorflow as tf

# Define a custom type to demonstrate the dispatch API
class MaskedTensor(tf.experimental.ExtensionType):
    values: tf.Tensor
    mask: tf.Tensor

# Leverage the similar API: tf.experimental.dispatch_for_unary_elementwise_apis
# This decorator overrides the default implementation for unary elementwise APIs
# (like tf.abs, tf.exp, etc.) when the input is of type MaskedTensor.
@tf.experimental.dispatch_for_unary_elementwise_apis(MaskedTensor)
def masked_unary_handler(api_func, x):
    """
    Handler for unary elementwise operations on MaskedTensor.
    This mimics the function definition pattern found in the original bug report.
    """
    # Apply the operation to the values, preserving the mask
    return MaskedTensor(api_func(x.values), x.mask)

# Reproduce the "compiled context" logic from the bug report
# The original bug used @torch.compile(fullgraph=True)
# Here we use @tf.function which is the TensorFlow equivalent for graph compilation.
@tf.function
def f(x):
    # In the original bug, torch._check was called inside the compiled function.
    # Here, we call a unary operation (tf.abs) which triggers the dispatch mechanism
    # defined above. This tests if the API integration works correctly within
    # a compiled graph without causing graph breaks or tracing errors.
    return tf.abs(x)

# Setup test data
# Original: x = torch.randn(3, device="cuda")
values = tf.constant([-1.0, -2.0, 3.0])
mask = tf.constant([True, False, True])
x = MaskedTensor(values, mask)

# Execute the compiled function
result = f(x)

# Assertions to verify correctness
# Check that the result is still a MaskedTensor
assert isinstance(result, MaskedTensor), "Result should be a MaskedTensor"

# Check if the operation (abs) was applied correctly to values
# abs([-1.0, -2.0, 3.0]) = [1.0, 2.0, 3.0]
expected_values = tf.constant([1.0, 2.0, 3.0])
assert tf.reduce_all(tf.equal(result.values, expected_values)), "Values were not transformed correctly"

# Check if mask is preserved
assert tf.reduce_all(tf.equal(result.mask, mask)), "Mask was not preserved"

print("Test passed: tf.experimental.dispatch_for_unary_elementwise_apis works correctly inside tf.function.")