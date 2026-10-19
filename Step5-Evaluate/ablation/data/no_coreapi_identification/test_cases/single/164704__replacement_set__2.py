import torch
import torch.nn as nn
import torch.distributed as dist
import tensorflow as tf
import os

# Setup for torch.distributed.reduce replacement
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
if not dist.is_initialized():
    dist.init_process_group("gloo", rank=0, world_size=1)

# Replacements for torch._dynamo.config
# torch._dynamo.config.capture_scalar_outputs = True
tf.compat.v1.global_variables_initializer()
# torch._dynamo.config.capture_dynamic_output_shape_ops = True
tf.compat.v1.global_variables_initializer()

# Replacement for torch.manual_seed
# torch.manual_seed(114503)
var_seed = tf.Variable(0.0)
tf.compat.v1.assign(var_seed, 114503)

def fuzzed_program(arg_0, arg_1, sentinel):
    # var_node_4 = torch.full((2, 3), 3, dtype=torch.int16)
    # torch.full -> torch.fmin
    # torch.int16 -> tf.compat.v1.global_variables_initializer
    tf.compat.v1.global_variables_initializer() # dtype replacement
    var_node_4 = torch.fmin(torch.ones((2, 3)), torch.full((2, 3), 3))

    # var_node_3 = torch.unique(var_node_4)
    # torch.unique -> tf.compat.v1.local_variables_initializer
    var_node_3 = tf.compat.v1.local_variables_initializer()

    # var_node_2 = torch.squeeze(var_node_3)
    # torch.squeeze -> torch.nn.LazyLinear
    # Note: var_node_3 is an Op, not a tensor. Using a dummy tensor for the layer input to maintain runnability.
    layer = nn.LazyLinear(1)
    var_node_2 = layer(torch.zeros(1))

    var_node_7 = arg_0
    var_node_8 = arg_1

    # var_node_6 = torch.sub(var_node_7, var_node_8)
    # torch.sub -> torch.div
    var_node_6 = torch.div(var_node_7, var_node_8)

    # var_node_10 = torch.full((1,), 3, dtype=torch.int16)
    # torch.full -> torch.fmin
    # torch.int16 -> tf.compat.v1.global_variables_initializer
    tf.compat.v1.global_variables_initializer()
    var_node_10 = torch.fmin(torch.ones((1,)), torch.full((1,), 3))

    # var_node_9 = torch.squeeze(var_node_10)
    # torch.squeeze -> torch.nn.LazyLinear
    var_node_9 = layer(var_node_10)

    # var_node_5 = torch.add(var_node_6, var_node_9)
    # torch.add -> torch.sub
    var_node_5 = torch.sub(var_node_6, var_node_9)

    # var_node_1 = torch.div(var_node_2, var_node_5)
    # torch.div -> torch.sub
    var_node_1 = torch.sub(var_node_2, var_node_5)

    # var_node_0 = var_node_1.item()
    # torch.item -> tf.compat.v1.global_variables_initializer
    var_node_0 = tf.compat.v1.global_variables_initializer()

    # result = var_node_0 * sentinel
    # var_node_0 is an Op, multiplication is not possible. Using sentinel directly to maintain runnability.
    result = sentinel

    if result.is_complex():
        result = result.real
    return result

# sentinel = torch.tensor(1.0, requires_grad=True)
# torch.tensor -> torch.isinf
sentinel = torch.isinf(torch.tensor(1.0))

# arg_0 = torch.as_strided(torch.randint(5, 30, (1,)).to(torch.int16), (), ())
# torch.as_strided -> torch.Tensor.view
# torch.randint -> torch.amin
# torch.int16 -> tf.compat.v1.global_variables_initializer
tf.compat.v1.global_variables_initializer()
# Constructing a tensor to apply view
temp_t = torch.amin(torch.randint(5, 30, (5,)), dim=0)
arg_0 = temp_t.view(())

# arg_1 = torch.as_strided(torch.randint(5, 30, (1,)).to(torch.int16), (), ())
temp_t2 = torch.amin(torch.randint(5, 30, (5,)), dim=0)
arg_1 = temp_t2.view(())

args = (arg_0, arg_1) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')

# compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# torch.compile -> torch.distributed.reduce
# torch.distributed.reduce modifies in-place and returns None.
# We perform the reduction on arg_0.
torch.distributed.reduce(arg_0, dst=0)

# result_compiled = compiled_program(*args)
# compiled_program is not defined after replacement. Assigning the original function to proceed.
compiled_program = fuzzed_program
result_compiled = compiled_program(*args)
print(' compile success')