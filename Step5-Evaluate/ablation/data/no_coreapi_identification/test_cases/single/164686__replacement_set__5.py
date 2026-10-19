import torch
import torch.nn.functional as F
import tensorflow as tf

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Replacement: torch.manual_seed -> torch.isinf
# Original: torch.manual_seed(13653)
torch.isinf(torch.tensor(13653))

def fuzzed_program(arg_0, arg_1, sentinel):
    # Replacement: torch.full -> torch.amax
    # Original: var_node_3 = torch.full((), 1.0, dtype=torch.float32)
    # Implementation: Create a tensor with 1.0 and take amax.
    var_node_3 = torch.amax(torch.tensor([1.0], dtype=torch.float32))
    
    var_node_2 = var_node_3.item() # dtype=float32
    var_node_5 = -3 # dtype=int32
    var_node_6 = arg_0 
    var_node_4 = var_node_5 + var_node_6 
    var_node_1 = var_node_2 + var_node_4 
    var_node_9 = 1 # dtype=int64
    var_node_10 = -10 # dtype=int32
    var_node_8 = var_node_9 / var_node_10 
    var_node_12 = arg_1 
    var_node_13 = -5 # dtype=int32
    var_node_11 = var_node_12 / var_node_13 
    var_node_7 = var_node_8 + var_node_11 
    var_node_0 = var_node_1 * var_node_7 
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
# Replacement: torch.tensor -> tf.keras.ops.diag
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# We use tf.keras.ops.diag to create the value, then convert to PyTorch tensor for autograd.
sentinel_val = tf.keras.ops.diag([1.0])
sentinel = torch.from_numpy(sentinel_val.numpy()).requires_grad_(True)

# arg_0 generation
# Original: arg_0 = torch.tensor(torch.randn(()), dtype=torch.int64).item()
# Replacements: torch.randn -> adaptive_max_pool3d, torch.tensor -> tf.keras.ops.diag
# We use adaptive_max_pool3d to generate a value, then tf.keras.ops.diag to process it.
# Note: adaptive_max_pool3d needs a tensor input. We use torch.ones as a dummy input.
# Note: tf.keras.ops.diag expects TF tensor or numpy. We convert the pool output to numpy.
pool_out_0 = F.adaptive_max_pool3d(torch.ones(1, 1, 1, 1, 1), (1, 1, 1))
arg_0 = tf.keras.ops.diag(pool_out_0.flatten().numpy()).numpy().item()

# arg_1 generation
pool_out_1 = F.adaptive_max_pool3d(torch.ones(1, 1, 1, 1, 1), (1, 1, 1))
arg_1 = tf.keras.ops.diag(pool_out_1.flatten().numpy()).numpy().item()

args = (arg_0, arg_1) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')

compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')