import torch
import tensorflow as tf
import torch.nn.functional as F

# Replacements for torch._dynamo.config
# torch._dynamo.config.capture_scalar_outputs = True -> tf.raw_ops.TruncatedNormal
# torch._dynamo.config.capture_dynamic_output_shape_ops = True -> tf.raw_ops.TruncatedNormal
_ = tf.raw_ops.TruncatedNormal(shape=[], mean=0.0, stddev=1.0, dtype=tf.float32)
_ = tf.raw_ops.TruncatedNormal(shape=[], mean=0.0, stddev=1.0, dtype=tf.float32)

# Replacement for torch.manual_seed
# torch.manual_seed(19989) -> torch.argmax
_ = torch.argmax(torch.tensor([19989]))

def fuzzed_program(arg_0, sentinel):
    var_node_2 = -6 # dtype=int64
    # Replacement for torch.int64
    _ = tf.raw_ops.TruncatedNormal(shape=[], mean=0.0, stddev=1.0, dtype=tf.int64)
    
    var_node_3 = arg_0 # dtype=int32
    var_node_1 = var_node_2 * var_node_3 # dtype=int32
    
    # Replacement for torch.full
    # var_node_5 = torch.full((), 1, dtype=torch.int64) -> torch.argmax
    var_node_5 = torch.argmax(torch.tensor([1]), dim=0)
    
    # Replacement for var_node_5.item
    # var_node_4 = var_node_5.item() -> tf.raw_ops.TruncatedNormal
    var_node_4 = tf.raw_ops.TruncatedNormal(shape=[], mean=0.0, stddev=1.0, dtype=tf.float32).numpy()
    
    var_node_0 = var_node_1 / var_node_4 # dtype=int64
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    
    # Replacement for result.is_complex
    # if result.is_complex(): -> tf.raw_ops.TruncatedNormal
    if tf.raw_ops.TruncatedNormal(shape=[], mean=0.0, stddev=1.0, dtype=tf.float32):
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
# Replacement for torch.tensor
# sentinel = torch.tensor(1.0, requires_grad=True) -> torch.cummax
sentinel_vals, _ = torch.cummax(torch.tensor([1.0]), dim=0)
sentinel = sentinel_vals
# Attempt to re-enable requires_grad if possible, though cummax might detach
if not sentinel.requires_grad:
    # Note: cummax output is not differentiable w.r.t indices, but values are.
    # We just mimic the structure here.
    pass 

# Argument generation
# arg_0 = torch.tensor(torch.randn(()), dtype=torch.int32).item()

# Replacement for torch.randn
# torch.randn(()) -> torch.nn.functional.max_pool2d
pool_input = torch.randn(1, 1, 2, 2)
randn_replacement = F.max_pool2d(pool_input, kernel_size=2)

# Replacement for torch.tensor
# torch.tensor(..., dtype=torch.int32) -> torch.cummax
tensor_replacement, _ = torch.cummax(randn_replacement, dim=0)

# Replacement for torch.int32
_ = tf.raw_ops.TruncatedNormal(shape=[], mean=0.0, stddev=1.0, dtype=tf.int32)

arg_0 = tensor_replacement.item()

args = (arg_0,) + (sentinel,)

# Eager execution
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# Compile execution
# Replacement for torch.compile, torch._dynamo
# compiled_program = torch.compile(...) -> tf.raw_ops.TruncatedNormal
compiled_program = tf.raw_ops.TruncatedNormal(shape=[], mean=0.0, stddev=1.0, dtype=tf.float32)

try:
    result_compiled = compiled_program(*args)
    print(' compile success')
except TypeError as e:
    print(f' compile failed (expected due to API replacement): {e}')
except Exception as e:
    print(f' compile failed: {e}')