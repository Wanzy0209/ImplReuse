import torch
import tensorflow as tf
import torch.library

# Replacement: torch.manual_seed -> tf.compat.v1.assign_add
# We create a TF variable and use assign_add to simulate the seed setup step
tf_seed_var = tf.Variable(0.0)
tf.compat.v1.assign_add(tf_seed_var, 974450504.0)

# Replacement: torch.tensor -> tf.experimental.numpy.diag
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# We create a diagonal matrix in TF and convert it to a PyTorch tensor
tf_sentinel = tf.experimental.numpy.diag([1.0])
sentinel = torch.from_numpy(tf_sentinel.numpy()).float().requires_grad_(True)

# Replacement: torch.as_strided -> torch.narrow
# Original: arg_0 = torch.as_strided(torch.randint(0, 2, (26010,), dtype=torch.int8).bool(), (17, 30, 17, 3), (1530, 51, 3, 1))
# We create a base tensor, reshape it, and use narrow to get a view
base_tensor = torch.randint(0, 2, (26010,), dtype=torch.int8).bool()
reshaped_base = base_tensor.view(17, 30, 17, 3)
arg_0 = torch.narrow(reshaped_base, 0, 0, 17)

# Replacement: torch.randint -> torch.fmax
# Original: arg_1 = torch.as_strided(torch.randint(5, 30, (15,)).to(torch.int64), (15,), (1,))
# We use fmax on two random tensors to generate arg_1
t1 = torch.randint(5, 30, (15,)).to(torch.int64)
t2 = torch.randint(5, 30, (15,)).to(torch.int64)
arg_1 = torch.fmax(t1, t2)

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0

    # Replacement: torch.chunk -> torch.index_select
    # Original: var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0]
    # chunk splits dim 3 (size 3) into 3 parts of size 1. [0] takes the first part.
    # We simulate this by selecting index 0 on dim 3.
    indices_chunk = torch.tensor([0], device=var_node_3.device)
    var_node_2 = torch.index_select(var_node_3, 3, indices_chunk)

    # Replacement: torch.full -> torch.amin
    # Original: var_node_5 = torch.full((17,), 3, dtype=torch.int64)
    # We use amin on var_node_3 to produce a tensor of shape (17,)
    var_node_5 = torch.amin(var_node_3, dim=(1, 2, 3)).to(torch.int64)

    var_node_6 = arg_1
    _input_size_var_node_4 = var_node_5.size(0)

    # Replacement: torch.randint -> torch.fmax
    # Original: _index_var_node_4 = torch.randint(0, _input_size_var_node_4, (15,), device=var_node_5.device)
    # This was used for gather. Since gather is replaced by ceil, we just need a valid fmax call.
    _index_var_node_4 = torch.fmax(torch.tensor(0, device=var_node_5.device), torch.tensor(_input_size_var_node_4 - 1, device=var_node_5.device))

    # Replacement: torch.gather -> torch.ceil
    # Original: var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4)
    # We apply ceil to var_node_5
    var_node_4 = torch.ceil(var_node_5.float())

    _input_size_var_node_1 = var_node_2.size(0)

    # Replacement: torch.randint -> torch.fmax
    # Original: _index_var_node_1 = torch.randint(0, _input_size_var_node_1, (15,), device=var_node_2.device)
    # This was used for index_select. Since index_select is replaced by cummin, we just need a valid fmax call.
    _index_var_node_1 = torch.fmax(torch.tensor(0, device=var_node_2.device), torch.tensor(_input_size_var_node_1 - 1, device=var_node_2.device))

    # Replacement: torch.index_select -> torch.cummin
    # Original: var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1)
    # We apply cummin on dim 0
    var_node_1 = torch.cummin(var_node_2, dim=0)[0]

    var_node_0 = torch.squeeze(var_node_1)

    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

args = (arg_0, arg_1, sentinel)
result_original = fuzzed_program(*args)
print(' eager success')

# Replacement: torch.compile -> torch.library.impl_abstract
# We define a custom library operation to replace the compilation mechanism.
# We register an abstract implementation and a concrete implementation.

def custom_op_meta(x, y, z):
    return torch.empty_like(x)

def custom_op_impl(x, y, z):
    return fuzzed_program(x, y, z)

# Define a unique namespace for the test to avoid conflicts
try:
    torch.library.define("test_api_replace::custom_run", "(Tensor, Tensor, Tensor) -> Tensor")
except AssertionError:
    pass # Op might already be defined

torch.library.impl_abstract("test_api_replace::custom_run", custom_op_meta)
torch.library.impl("test_api_replace::custom_run", custom_op_impl)

# Use the custom op instead of torch.compile
compiled_program = torch.ops.test_api_replace.custom_run

result_compiled = compiled_program(*args)
print(' compile success')