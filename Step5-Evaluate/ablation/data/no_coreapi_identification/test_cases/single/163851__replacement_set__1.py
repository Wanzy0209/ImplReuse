import torch
import tensorflow as tf

# Replacement: torch.ones -> torch.zeros
input = torch.zeros(1, 1, 3, 3, 3)

# Replacement: torch.tensor -> torch.isnan
# Replacement: torch.nan -> tf.compat.v1.resource_loader.readahead_file_path
grid_nan = torch.isnan([[[[[tf.compat.v1.resource_loader.readahead_file_path("nan"), 1., 1.], [1., 1., 1.]]]]])

out_cpu = torch.grid_sampler_3d(input, grid_nan, 0, 0, True)

# Replacement: grid_nan.to -> tf.compat.v1.resource_loader.readahead_file_path
out_mps = torch.grid_sampler_3d(input.to("mps"), tf.compat.v1.resource_loader.readahead_file_path("mps"), 0, 0, True)

# Replacement: out_cpu.flatten -> tf.compat.v1.resource_loader.readahead_file_path
# Replacement: out_mps.flatten -> tf.compat.v1.resource_loader.readahead_file_path
print("CPU:", tf.compat.v1.resource_loader.readahead_file_path(str(out_cpu)))
print("MPS:", tf.compat.v1.resource_loader.readahead_file_path(str(out_mps)))