import torch
import tensorflow as tf

# Replacement: torch.tensor -> tf.linalg.diag
# Original input was [1, -3, 5]. 
# tf.linalg.diag creates a diagonal matrix using these values.
tf_input = tf.linalg.diag([1, -3, 5])

# Convert the TensorFlow tensor to a PyTorch tensor
torch_input = torch.from_numpy(tf_input.numpy())

# Call the similar API: torch.min
# Original API was torch.aminmax, which returns a named tuple (min, max).
# torch.min returns the minimum value (scalar) when no dimension is specified.
result = torch.min(torch_input)

# Assertion
# The minimum value among 1, -3, and 5 is -3.
assert result.item() == -3