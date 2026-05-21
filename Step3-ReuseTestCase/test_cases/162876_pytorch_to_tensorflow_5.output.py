import torch
import tensorflow as tf
import tf.experimental.numpy as tnp

# Adapted test case for tf.experimental.numpy.vdot
# The original issue involved torch.aminmax which returns a named tuple.
# tf.experimental.numpy.vdot returns a scalar (ndarray), so we test the functional call.
# We use two input vectors to mimic the input data style.

a = tnp.array([1, -3])
b = tnp.array([5, 2])

# Call the API
result = tnp.vdot(a, b)
print(result)

# Verify the result: 1*5 + (-3)*2 = -1
assert result == -1