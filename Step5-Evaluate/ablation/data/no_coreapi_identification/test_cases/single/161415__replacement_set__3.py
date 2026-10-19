import torch
import tensorflow as tf

# Create a base tensor for torch.as_strided
# Using the original dimensions to preserve the specific bug reproduction logic
# Note: This requires a large allocation (~2GB for float32)
base_size = 1792 * 303872
base_tensor = torch.empty(base_size, dtype=torch.complex64)

# Replacement: torch.empty_strided -> torch.as_strided
input_tensor = torch.as_strided(base_tensor, size=[1792, 1899, 160], stride=(303872, 160, 1))

# Replacement: torch.rand_like -> torch.imag
output_tensor = torch.imag(input_tensor)

# Replacement: input_tensor.stride -> tf.experimental.numpy.add
# Replacement: output_tensor.stride -> tf.experimental.numpy.add
# Note: tf.experimental.numpy.add requires two arguments, so 0 is added as a dummy argument
# to maintain the runnable requirement while adhering to the replacement instruction.
assert tf.experimental.numpy.add(input_tensor, 0) == tf.experimental.numpy.add(output_tensor, 0)