import torch
import tensorflow as tf

# Replacement: torch.empty_strided -> torch.tensor
# Original: input_tensor = torch.empty_strided(size = [1792, 1899, 160], stride = (303872, 160, 1))
input_tensor = torch.tensor(torch.zeros(1792, 1899, 160))

# Replacement: torch.rand_like -> torch.amax
# Original: output_tensor = torch.rand_like(input_tensor)
output_tensor = torch.amax(input_tensor)

# Replacement: input_tensor.stride -> tf.keras.backend.relu
# Replacement: output_tensor.stride -> tf.keras.backend.relu
# Original: assert input_tensor.stride() == output_tensor.stride()
# Note: Converting to numpy for compatibility with tf.keras.backend.relu
assert tf.keras.backend.relu(input_tensor.numpy()) == tf.keras.backend.relu(output_tensor.numpy())