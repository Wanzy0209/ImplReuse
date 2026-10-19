import torch
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

# torch._dynamo.config.capture_scalar_outputs = True

# torch.manual_seed -> torch.isinf
torch.isinf(torch.tensor(1.0))

def fuzzed_program(arg_0, sentinel):
    var_node_4 = arg_0 # size=(1, 2, 2), dtype=float, device=cpu (from torch.bmm)
    
    # torch.chunk -> torch.nn.functional.pdist
    # pdist expects 2D input. arg_0 is 3D (1, 2, 2). We squeeze the batch dim.
    # Original chunk took size 4, returned size 1.
    # pdist on (2, 2) returns size 1.
    var_node_3 = F.pdist(var_node_4.squeeze(0))
    
    # torch.squeeze (Target API)
    var_node_2 = torch.squeeze(var_node_3)
    
    # torch.stack -> torch.zeros_like
    var_node_1 = torch.zeros_like(var_node_2)
    
    # torch.reshape -> tf.compat.v1.gradients
    # Convert PyTorch tensor to TensorFlow tensor
    tf_var_1 = tf.constant(var_node_1.numpy())
    
    # Use tf.compat.v1.gradients. It works best in graph mode or with tf.function.
    @tf.function
    def compute_grads(x):
        return tf.compat.v1.gradients(x, x)[0]
        
    var_node_0 = compute_grads(tf_var_1)
    
    # result = var_node_0 * sentinel
    # sentinel is a TF tensor. var_node_0 is a TF tensor.
    result = var_node_0 * sentinel
    
    # if result.is_complex():
    if result.dtype == tf.complex64 or result.dtype == tf.complex128:
        # result = result.real
        result = tf.math.real(result)
        
    # Convert back to PyTorch for torch.all
    return torch.from_numpy(result.numpy())

# torch.tensor -> tf.keras.ops.diag
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# New: sentinel = tf.keras.ops.diag([1.0])
sentinel = tf.keras.ops.diag([1.0])

# torch.as_strided -> torch.bmm
# torch.randint -> torch.full
# Original: arg_0 = torch.as_strided(torch.randint(0, 2, (4,), dtype=torch.int8).bool(), (4,), (1,))
# New: Create inputs for bmm.
# bmm needs (B, M, K) and (B, K, N).
# Let's make B=1, M=2, K=2, N=2.
t1 = torch.full((1, 2, 2), 1.0)
t2 = torch.full((1, 2, 2), 1.0)
arg_0 = torch.bmm(t1, t2)

args = (arg_0,) + (sentinel,)

# Run original (eager)
result_original = fuzzed_program(*args)
print(' eager success')

# torch.compile -> torch.all
# Original: compiled_program = torch.compile(...)
# result_compiled = compiled_program(...)
# New: result_compiled = torch.all(fuzzed_program(...))
result_compiled = torch.all(fuzzed_program(*args))
print(' compile success')