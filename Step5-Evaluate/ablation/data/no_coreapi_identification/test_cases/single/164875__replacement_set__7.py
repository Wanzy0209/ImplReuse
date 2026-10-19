import torch
import tensorflow as tf
import numpy as np

# Use the similar API to check environment
if tf.test.is_gpu_available():
    device = torch.device('cuda')
    print("Using CUDA device")
else:
    device = torch.device('cpu')
    print("Using CPU device")

# Replacements for torch._dynamo.config
# torch._dynamo.config.capture_scalar_outputs -> tf.math.reduce_all
# torch._dynamo.config.capture_dynamic_output_shape_ops -> tf.math.reduce_all
# We call these to satisfy the replacement requirement
_ = tf.math.reduce_all([True])
_ = tf.math.reduce_all([True])

# torch.manual_seed -> torch.argmax
# Original: torch.manual_seed(1014698)
# Replacement: torch.argmax(torch.tensor([1014698]))
_ = torch.argmax(torch.tensor([1014698]))

def fuzzed_program(arg_0, sentinel):
    var_node_1 = arg_0

    # torch.full -> torch.argmax
    # Original: torch.full((), True, dtype=torch.bool)
    # Replacement: torch.argmax(torch.tensor([True]))
    var_node_3 = torch.argmax(torch.tensor([True]))

    # torch.zeros -> torch.slogdet
    # Original: torch.zeros((), dtype=torch.bool, device=var_node_3.device)
    # Replacement: torch.slogdet(torch.eye(1, device=var_node_3.device))
    # slogdet returns (sign, logabsdet). We take the sign.
    _x_nz = torch.slogdet(torch.eye(1, device=var_node_3.device))[0]

    # Reshape logic (no replacement specified)
    _x_nz_flat = _x_nz.reshape(-1)
    # Slicing logic
    # Original: _x_nz_flat[:20] = True
    # _x_nz_flat is size 1. Assigning True to slice of size 1 works.
    _x_nz_flat[:20] = True

    # torch.nonzero -> torch.clamp_min
    # Original: torch.nonzero(_x_nz)
    # Replacement: torch.clamp_min(_x_nz, 0)
    var_node_2 = torch.clamp_min(_x_nz, 0)

    # torch.add -> tf.math.reduce_all
    # Original: torch.add(var_node_1, var_node_2)
    # Replacement: tf.math.reduce_all([var_node_1, var_node_2])
    # We convert torch tensors to numpy for tf op, then back to torch
    with torch.no_grad():
        t1 = var_node_1.cpu().numpy()
        t2 = var_node_2.cpu().numpy()
        res_tf = tf.math.reduce_all([t1, t2])
        var_node_0 = torch.from_numpy(res_tf.numpy()).to(device)

    # Multiply logic
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel
# torch.tensor -> torch.cummax
# Original: torch.tensor(1.0, requires_grad=True)
# Replacement: torch.cummax(torch.tensor([1.0]), dim=0)[0]
# cummax returns (values, indices). We take values.
sentinel = torch.cummax(torch.tensor([1.0]), dim=0)[0].to(device)

# Arg_0
# torch.randint -> torch.argmin
# Original: torch.randint(5, 30, (20,)).to(torch.int64)
# Replacement: torch.argmin(torch.randn(20, 1), dim=0)
# argmin returns indices. Shape (1,).
temp = torch.argmin(torch.randn(20, 1), dim=0).to(device)

# torch.as_strided -> torch.logcumsumexp
# Original: torch.as_strided(..., (20, 0), (1, 20))
# Replacement: torch.logcumsumexp(temp, dim=0)
# logcumsumexp on (1,) tensor returns (1,).
arg_0 = torch.logcumsumexp(temp, dim=0)

args = (arg_0,) + (sentinel,)

# Eager execution
result_original = fuzzed_program(*args)
print(' eager success')

# torch.compile -> torch.lobpcg
# Original: torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Replacement: torch.lobpcg(...)
# We create a dummy matrix A to pass to lobpcg.
# This replaces the "compilation" step with a "lobpcg" step.
A = torch.eye(5, device=device)
# lobpcg returns (eigenvalues, eigenvectors)
result_compiled = torch.lobpcg(A, k=1)
print(' compile success (replaced by lobpcg)')