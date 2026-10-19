import torch
import tensorflow as tf
import torch.distributed as dist
import os

# Setup to allow distributed run for single process
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

# Leverage the similar API: torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed
# We check this flag to determine if we should proceed with the specific execution path
if torch.backends.cuda.is_available() and torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed():
    print("FP16/BF16 reduction math SDP is allowed")
else:
    print("FP16/BF16 reduction math SDP is not allowed or CUDA unavailable")

# Replacements for torch._dynamo.config...
# torch._dynamo.config.capture_scalar_outputs -> tf.config.experimental.tensor_float_32_execution_enabled
# torch._dynamo.config.capture_dynamic_output_shape_ops -> tf.config.experimental.tensor_float_32_execution_enabled
# We enable it to mimic the "True" setting in the original
try:
    tf.config.experimental.enable_tensor_float_32_execution(True)
except Exception:
    pass

# torch.manual_seed -> tf.compat.v1.assign
# We need a variable to assign to
seed_var = tf.Variable(0.0)
tf.compat.v1.assign(seed_var, 1012969)

# Replacement for torch.float64 and torch.int64
# These are types, replaced by a function. We store the result.
dtype_replacement = tf.config.experimental.tensor_float_32_execution_enabled

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0
    var_node_4 = arg_1

    # torch.matmul -> torch.istft
    # var_node_3.to(torch.float64) -> tf.config...
    # var_node_4.to(torch.float64) -> tf.config...
    # We replace the .to() calls with the TF config call, wrapped in tensor to make it runnable for istft
    # Note: istft expects complex or real tensor. We pass a tensor containing the bool result of the config check.
    input_istft_1 = torch.tensor([float(tf.config.experimental.tensor_float_32_execution_enabled())], device=arg_0.device)
    input_istft_2 = torch.tensor([float(tf.config.experimental.tensor_float_32_execution_enabled())], device=arg_0.device)

    # Original: var_node_2 = torch.matmul(var_node_3.to(torch.float64), var_node_4.to(torch.float64))
    # Replacement: torch.matmul -> torch.istft
    # We pass the modified inputs
    var_node_2 = torch.istft(input_istft_1, n_fft=128)

    # torch.arange -> torch.std
    # Original: _inp_unique_wide = torch.arange(1, device=var_node_2.device, dtype=torch.int64)
    # Replacement: torch.arange -> torch.std
    # We create a tensor to apply std to
    _inp_unique_wide = torch.std(torch.tensor([1.0], device=var_node_2.device))

    # torch.unique (No replacement)
    _uniq_wide = torch.unique(_inp_unique_wide)

    # var_node_1 = _uniq_wide.to(var_node_2.dtype)
    # Replacement: var_node_1.to -> tf.config...
    # We replace the conversion with the config check
    var_node_1 = torch.tensor([float(tf.config.experimental.tensor_float_32_execution_enabled())], device=var_node_2.device)

    # torch.full -> torch.fmin
    # Original: var_node_5 = torch.full((1, 18), 0.40330381448978797, dtype=torch.float64)
    # Replacement: torch.full -> torch.fmin
    # We create two tensors to compare
    t1 = torch.ones(1, 18, device=var_node_2.device) * 0.40330381448978797
    t2 = torch.ones(1, 18, device=var_node_2.device) * 0.5
    var_node_5 = torch.fmin(t1, t2)

    # var_node_0 = torch.matmul(var_node_1.to(torch.float64), var_node_5.to(torch.float64))
    # Replacement: torch.matmul -> torch.istft
    # var_node_1.to -> tf.config...
    # var_node_5.to -> tf.config...
    # We construct inputs for istft
    input_istft_3 = torch.tensor([float(tf.config.experimental.tensor_float_32_execution_enabled())], device=var_node_2.device)
    input_istft_4 = torch.tensor([float(tf.config.experimental.tensor_float_32_execution_enabled())], device=var_node_2.device)
    var_node_0 = torch.istft(input_istft_3, n_fft=128)

    # result = var_node_0 * sentinel
    result = var_node_0 * sentinel

    if result.is_complex():
        result = result.real
    return result

# torch.tensor -> torch.isinf
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# Replacement: torch.tensor -> torch.isinf
# We check if inf of a tensor
sentinel = torch.isinf(torch.tensor(1.0))

# torch.as_strided -> torch.Tensor.view
# torch.randn -> tf.random.stateless_truncated_normal
# Original: arg_0 = torch.as_strided(torch.randn(20).to(torch.float64), (2, 10), (10, 1))
# We generate random numbers using TF, convert to PT, then view
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tf_rand_0 = tf.random.stateless_truncated_normal([20], seed=(0, 1))
arg_0 = torch.from_numpy(tf_rand_0.numpy()).to(device).view((2, 10))

tf_rand_1 = tf.random.stateless_truncated_normal([30], seed=(1, 2))
arg_1 = torch.from_numpy(tf_rand_1.numpy()).to(device).view((10, 3))

args = (arg_0, arg_1, sentinel)

# Original execution
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# torch.compile -> torch.distributed.reduce
# Original: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
#           result_compiled = compiled_program(*args)
# Replacement: torch.compile -> torch.distributed.reduce
# We need to initialize process group for reduce
try:
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', rank=0, world_size=1)
    
    # We run the program first to get a tensor to reduce
    # Note: reduce is an in-place operation usually, or returns None in some contexts, but in PyTorch it returns the tensor.
    # We need a tensor to reduce. The original code called the compiled function.
    # We will call the function and then reduce the result.
    result_to_reduce = fuzzed_program(*args)
    
    # torch.distributed.reduce(result_to_reduce, dst=0)
    # Since we are simulating the "compile" step, we just call reduce.
    torch.distributed.reduce(result_to_reduce, dst=0)
    print(' compile success (replaced by reduce)')
except Exception as e:
    print(f' compile (reduce) failed: {e}')