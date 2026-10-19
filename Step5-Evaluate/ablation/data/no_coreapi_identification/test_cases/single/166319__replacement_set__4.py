import torch
import torch.fx
import tensorflow as tf

torch._dynamo.config.capture_scalar_outputs = True

# Replacement for torch.manual_seed
# Original: torch.manual_seed(1215252001)
# Replacement: tf.compat.v1.assign_add
# We need a variable to assign to.
_seed_var = tf.Variable(0, dtype=tf.int32)
tf.compat.v1.assign_add(_seed_var, 1215252001)

def fuzzed_program(arg_0, arg_1, arg_2, sentinel):
    # var_node_4 = arg_0
    # Replacement for torch.squeeze
    # Original: var_node_3 = torch.squeeze(var_node_4)
    # Replacement: torch.fx.wrap
    # To make it runnable and return a tensor, we wrap an identity function
    var_node_3 = torch.fx.wrap(lambda x: x)(arg_0)

    # Replacement for torch.chunk
    # Original: var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0]
    # Replacement: torch.index_select
    # Selecting the first few indices to simulate the first chunk
    # Assuming dim 2 has size 4 (from original size (9, 15, 4)), chunk size is 1.
    # Original size was (9, 15, 4). chunk(..., 4, dim=2) -> chunks of size 1. [0] is index 0.
    var_node_2 = torch.index_select(var_node_3, 2, torch.tensor([0], device=var_node_3.device))

    # Replacement for torch.squeeze
    # Original: var_node_1 = torch.squeeze(var_node_2)
    # Replacement: torch.fx.wrap
    var_node_1 = torch.fx.wrap(lambda x: x)(var_node_2)

    var_node_7 = arg_1
    var_node_8 = arg_2
    _input_size_var_node_6 = var_node_7.size(0)

    # Replacement for torch.randint
    # Original: _index_var_node_6 = torch.randint(0, _input_size_var_node_6, (18, 15), device=var_node_7.device)
    # Replacement: torch.fmax
    # fmax takes two tensors. We create tensors of the bounds.
    low = torch.full((18, 15), 0, device=var_node_7.device, dtype=torch.int64)
    high = torch.full((18, 15), _input_size_var_node_6, device=var_node_7.device, dtype=torch.int64)
    _index_var_node_6 = torch.fmax(low, high)

    # Replacement for torch.gather
    # Original: var_node_6 = torch.gather(var_node_7, 0, _index_var_node_6)
    # Replacement: torch.ceil
    var_node_6 = torch.ceil(var_node_7.to(torch.float)) # ceil requires float

    # Replacement for torch.chunk
    # Original: var_node_5 = torch.chunk(var_node_6, 2, dim=0)[0]
    # Replacement: torch.index_select
    # Size (18, 15), chunk 2 -> size 9. Select indices 0-8.
    var_node_5 = torch.index_select(var_node_6, 0, torch.arange(9, device=var_node_6.device))

    # Replacement for torch.mul
    # Original: var_node_0 = torch.mul(var_node_1, var_node_5)
    # Replacement: torch.square
    var_node_0 = torch.square(var_node_1.to(torch.float)) # square requires float usually, or handles int

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
# Replacement for torch.tensor
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# Replacement: tf.experimental.numpy.diag
# We create the TF tensor then convert to PyTorch to maintain compatibility with the rest of the script
tf_sentinel = tf.experimental.numpy.diag([1.0])
sentinel = torch.from_numpy(tf_sentinel.numpy()).to(torch.float32)
sentinel.requires_grad = True

# Replacement for torch.as_strided
# Original: arg_0 = torch.as_strided(torch.randint(5, 30, (484,)).to(torch.int32), (9, 1, 15, 4), (60, 60, 0, 1))
# Replacement: torch.narrow
base_0 = torch.randint(5, 30, (484,)).to(torch.int32)
arg_0 = torch.narrow(base_0, 0, 0, 9 * 1 * 15 * 4).view(9, 1, 15, 4) # Approximation to get size right

# Original: arg_1 = torch.as_strided(torch.randint(5, 30, (300,)).to(torch.int64), (20, 15), (15, 1))
# Replacement: torch.narrow
base_1 = torch.randint(5, 30, (300,)).to(torch.int64)
arg_1 = torch.narrow(base_1, 0, 0, 20 * 15).view(20, 15)

# Original: arg_2 = torch.as_strided(torch.randint(5, 30, (270,)).to(torch.int64), (18, 15), (15, 1))
# Replacement: torch.narrow
base_2 = torch.randint(5, 30, (270,)).to(torch.int64)
arg_2 = torch.narrow(base_2, 0, 0, 18 * 15).view(18, 15)

args = (arg_0, arg_1, arg_2) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')