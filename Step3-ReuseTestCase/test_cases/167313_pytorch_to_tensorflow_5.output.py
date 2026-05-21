import torch
import tensorflow as tf
import numpy as np

# Adapted test case for tf.experimental.numpy.iscomplexobj
# Original Bug: torch.compile ignored alpha/beta parameters in addmm.
# Adaptation: Verify that tf.function (TF's compilation) correctly handles
# the type checking logic of iscomplexobj, ensuring no behavior change
# between eager and compiled execution.

# Setup inputs
# We use both real and complex tensors to verify the API's core logic
x_real = tf.constant([1.0, 2.0, 3.0])
x_complex = tf.constant([1.0 + 2.0j, 3.0 + 4.0j])

# Define the function using the target API
f = lambda x: tf.experimental.numpy.iscomplexobj(x)

# Define the compiled version (equivalent to torch.compile)
fc = tf.function(f)

# Test Real Tensor
print("Testing Real Tensor:")
out_eager = f(x_real)
out_compiled = fc(x_real)
print(f"Eager result: {out_eager}")
print(f"Compiled result: {out_compiled}")
assert out_eager == False, "Eager execution should return False for real tensor"
assert out_compiled == False, "Compiled execution should return False for real tensor"

# Test Complex Tensor
print("\nTesting Complex Tensor:")
out_eager = f(x_complex)
out_compiled = fc(x_complex)
print(f"Eager result: {out_eager}")
print(f"Compiled result: {out_compiled}")
assert out_eager == True, "Eager execution should return True for complex tensor"
assert out_compiled == True, "Compiled execution should return True for complex tensor"

print("\nTest passed: Behavior is consistent between eager and compiled execution.")