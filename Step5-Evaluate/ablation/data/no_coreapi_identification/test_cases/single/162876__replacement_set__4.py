import torch
import tensorflow as tf

# Replacement: torch.tensor -> tf.experimental.numpy.diag
# Original: torch.tensor([1, -3, 5])
# New: tf.experimental.numpy.diag([1, -3, 5])
# We convert the result to a torch tensor to be used by torch.min
input_tensor = torch.from_numpy(tf.experimental.numpy.diag([1, -3, 5]).numpy())

# Original: torch.aminmax(...)
# Similar API: torch.min(..., dim=0) to get a named tuple return type
torch.min(input_tensor, dim=0)

# Original: torch.return_types.aminmax(...)
# Similar API: torch.return_types.min(...)
# Attempting to construct the return type manually (mimicking the original bug)
torch.return_types.min(
    values=torch.tensor([0, -3, 0]),
    indices=torch.tensor([1, 1, 0])
)