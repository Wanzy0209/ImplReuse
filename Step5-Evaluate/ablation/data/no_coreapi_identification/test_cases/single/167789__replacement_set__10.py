import sys
import torch
import tensorflow as tf

def fn(x, n):
    if n == 0:
        return x
    return fn(x, n - 1) + 1

@torch.compile(backend="eager")
def outer(x):
    return fn(x, 1000)

# Replacement: sys.setrecursionlimit -> tf.compat.v1.linalg.trace
# Original: sys.setrecursionlimit(10000000)
# Adaptation: tf.compat.v1.linalg.trace requires a tensor input.
# We pass a tensor representing the limit value to satisfy the API signature.
tf.compat.v1.linalg.trace(tf.constant([[10000000]]))

# Replacement: torch.ones -> torch.qr
# Original: outer(torch.ones(3))
# Adaptation: torch.qr requires a matrix (2D tensor) and returns a tuple (Q, R).
# We create a matrix input and extract Q to pass to outer, ensuring the code is runnable.
input_matrix = torch.tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
q, _ = torch.qr(input_matrix)
outer(q)