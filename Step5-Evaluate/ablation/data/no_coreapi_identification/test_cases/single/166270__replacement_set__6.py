import torch
import tensorflow as tf
import numpy as np

# torch.manual_seed -> torch.isnan
torch.isnan(torch.tensor(0.0))

# torch.tensor -> tf.linalg.diag
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# Replacement:
tf_sentinel = tf.linalg.diag([1.0])
sentinel = torch.from_numpy(tf_sentinel.numpy())

def fuzzed_program(arg_0, sentinel):
    var_node_4 = arg_0 # size=(4,)

    # torch.chunk -> torch.triu
    # Original: var_node_3 = torch.chunk(var_node_4, 4, dim=0)[0] # size=(1,)
    # Replacement: torch.triu on 1D tensor returns the tensor.
    var_node_3 = torch.triu(var_node_4) # size=(4,)

    # API Under Test: torch.squeeze
    # Original: var_node_2 = torch.squeeze(var_node_3) # size=()
    # With replacement: var_node_3 is (4,), squeeze does nothing. size=(4,)
    var_node_2 = torch.squeeze(var_node_3)

    # torch.stack -> torch.baddbmm
    # Original: var_node_1 = torch.stack([var_node_2], dim=0) # size=(1,)
    # Replacement: baddbmm requires 3D inputs.
    # var_node_2 is (4,). Reshape to (1, 1, 4) and (1, 4, 1) to get (1, 1, 1) output.
    batch = torch.zeros(1, 1, 1)
    mat1 = var_node_2.reshape(1, 1, 4).float()
    mat2 = var_node_2.reshape(1, 4, 1).float()
    var_node_1 = torch.baddbmm(batch, mat1, mat2) # size=(1, 1, 1)

    # torch.reshape -> tf.gradients
    # Original: var_node_0 = torch.reshape(var_node_1, [1]) # size=(1,)
    # Replacement:
    tf_var = tf.convert_to_tensor(var_node_1.numpy())
    grads = tf.gradients(tf_var, tf_var)[0]
    var_node_0 = torch.from_numpy(grads.numpy()) # size=(1, 1, 1)

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# torch.randint -> torch.isreal
# torch.as_strided -> torch.empty
# Original: arg_0 = torch.as_strided(torch.randint(0, 2, (4,), dtype=torch.int8).bool(), (4,), (1,))
# Replacements:
_ = torch.isreal(torch.tensor(1.0)) # Replaces torch.randint
arg_0 = torch.empty((4,), dtype=torch.bool) # Replaces torch.as_strided

args = (arg_0, sentinel)
result_original = fuzzed_program(*args)
print(' eager success')

# torch.compile -> torch.any
# Original: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Replacement:
compiled_program = torch.any

# Original: result_compiled = compiled_program(*args)
# Adaptation: Pass the result of the function to torch.any
result_compiled = compiled_program(fuzzed_program(*args))
print(' compile success')