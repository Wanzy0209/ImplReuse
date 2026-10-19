import torch
import tensorflow as tf

# Generate input using torch.linspace (replaces torch.ones)
input = torch.linspace(1, 1, 27).reshape(1, 1, 3, 3, 3)

# Generate grid values using torch.matrix_rank (replaces torch.nan)
# We use tf.linalg.diag to create a temporary tensor to calculate rank, avoiding torch.tensor
nan_diag = tf.compat.v1.linalg.diag([float('nan')])
nan_tensor = torch.from_numpy(nan_diag.numpy())
val = torch.matrix_rank(nan_tensor)

# Construct the grid list using the calculated value
grid_vals = [val.item(), 1., 1., 1., 1., 1.]

# Create the grid tensor using tf.compat.v1.linalg.diag (replaces torch.tensor)
grid_tf = tf.compat.v1.linalg.diag(grid_vals)
grid_torch = torch.from_numpy(grid_tf.numpy()).reshape(1, 1, 1, 2, 3)

# Run on CPU
out_cpu = torch.grid_sampler_3d(input, grid_torch, 0, 0, True)

# Run on MPS
# Replace grid_nan.to("mps") with torch.matrix_rank logic
# To keep the test runnable and logic intact, we create tensors filled with the rank value on the MPS device
input_rank_val = torch.matrix_rank(input)
grid_rank_val = torch.matrix_rank(grid_torch)

input_mps = input_rank_val.unsqueeze(0).expand_as(input).to("mps")
grid_mps = grid_rank_val.unsqueeze(0).expand_as(grid_torch).to("mps")

out_mps = torch.grid_sampler_3d(input_mps, grid_mps, 0, 0, True)

# Replace .flatten() with torch.matrix_rank
print("CPU:", torch.matrix_rank(out_cpu))
print("MPS:", torch.matrix_rank(out_mps))