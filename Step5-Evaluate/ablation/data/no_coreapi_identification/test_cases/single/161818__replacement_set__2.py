import tensorflow as tf
import torch
from torch import nn

# Leverage the similar API (torch.nn.Sigmoid) as a candidate for reuse
# instead of nn.Linear to test the bug reproduction logic.
module = nn.Sigmoid()

# torch.rand is kept to ensure the test case is runnable and generates data.
# The replacement tf.types.experimental.distributed.Mirrored is a TensorFlow type
# and cannot generate PyTorch tensors.
padded = torch.rand(9, 8)

# Replacement: torch.as_tensor -> torch.atleast_3d
lengths = torch.atleast_3d([5, 4])

# torch.autograd.set_detect_anomaly is kept to detect the error in the backward pass.
# The replacement tf.types.experimental.distributed.Mirrored is not a valid context manager here.
with torch.autograd.set_detect_anomaly(True):
    out = module(padded)
    
    # torch.nested.narrow is kept as it is the core API under test for the bug.
    # The replacement tf.types.experimental.distributed.Mirrored is not a valid substitute for the operation.
    nopad = torch.nested.narrow(out, dim=1, start=0, length=lengths, layout=torch.jagged).contiguous().values()
    
    nopad.sum().backward()

# Replacement: torch.tensor -> torch.isinf
# Using torch.isinf to check the output as a sanity check, 
# mirroring the intent of creating/checking tensor properties.
assert not torch.isinf(out).any(), "Output contains infinity values"