import torch
import torch.nn as nn
import torch.nn.utils as utils

# Create a Linear module
m = nn.Linear(5, 7)

# Apply spectral normalization (analogous to prune.random_unstructured in the original example)
m = utils.spectral_norm(m, name='weight')

# Remove the spectral normalization reparameterization
m = utils.remove_spectral_norm(m, name='weight')

# Verify that the reparameterization is removed
# Spectral norm introduces 'weight_g' and 'weight_v' buffers; removal should delete them
assert not hasattr(m, 'weight_g')
assert not hasattr(m, 'weight_v')
assert isinstance(m.weight, torch.nn.Parameter)