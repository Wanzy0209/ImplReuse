import torch
import torch.nn.functional as F
import tensorflow as tf

# Replacing torch.randn with torch.nn.functional.adaptive_max_pool2d
# We create a dummy input tensor to apply the pooling operation.
# Original shape (10, 20, 30) is treated as (C, H, W) for pooling.
dummy_input = torch.ones(10, 20, 30, device='cuda')
x = F.adaptive_max_pool2d(dummy_input, (20, 30))

# Replacing torch.tensor with tf.experimental.numpy.diag
# We create the value using TensorFlow's numpy compatibility and convert to PyTorch.
tf_val = tf.experimental.numpy.diag([5.0])
max_val = torch.from_numpy(tf_val.numpy()).to('cuda')

# Using the similar API: torch.nn.functional.tanh
# Preserving the original bug reproduction logic: using .item() on a float tensor arg
def f(x, max_val):
    # Original logic used torch.clamp(x, 0, max_val.item())
    # We replace torch.clamp with torch.nn.functional.tanh (Similar API)
    # while maintaining the usage of .item() on the argument.
    y = F.tanh(x * max_val.item())
    return y

# torch.compile is used to trigger the Inductor backend
compiled_func = torch.compile(f, backend='inductor', fullgraph=True)

# Execute the compiled function
# This should trigger the NameError if the bug is present
result = compiled_func(x, max_val)

# Assertion to verify execution
assert result.shape == x.shape
print("Test passed.")