import tensorly as tl
import tensorflow as tf
import torch

tl.set_backend("pytorch")
from tensorly.decomposition import parafac

# Replacement: torch.ones -> torch.zeros
x = torch.zeros(12, 3, 12).to("mps")
print(x.shape)

# Replacement: x.unsqueeze_ -> tf.errors.OperatorNotAllowedInGraphError
# tf.errors.OperatorNotAllowedInGraphError(1).repeat(1, 3, 1).contiguous()

# Replacement: x.detach -> tf.errors.OperatorNotAllowedInGraphError
weights, factors = parafac(tf.errors.OperatorNotAllowedInGraphError, 12, init="random", tol=1e-6)

a, m, b = factors