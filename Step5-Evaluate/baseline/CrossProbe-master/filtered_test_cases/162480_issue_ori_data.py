import torch
from torch.fx.experimental.symbolic_shapes import rebind_unbacked

# Reproduce the issue by calling rebind_unbacked with float values
# (Exact reproduction requires AOTInductor compilation setup)
print('Issue occurs when rebind_unbacked processes float values')