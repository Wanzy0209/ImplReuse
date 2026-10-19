import torch
import tensorflow as tf
import torch.distributed as dist
import os

# --- Replacements for Setup and Config ---

# torch._dynamo.config.capture_scalar_outputs -> torch.library.register_kernel
# torch._dynamo.config.capture_dynamic_output_shape_ops -> torch.library.register_kernel
def dummy_config_op(x): return x
torch.library.register_kernel("test_ns", "capture_scalar_outputs", dummy_config_op)
torch.library.register_kernel("test_ns", "capture_dynamic_output_shape_ops", dummy_config_op)

# torch.manual_seed -> tf.compat.v1.assert_variables_initialized
try:
    # In TF 2.x, this might not be strictly necessary without sessions, but we follow the replacement.
    tf.compat.v1.assert_variables_initialized()
except Exception:
    pass

# Leverage similar API: tf.experimental.dtensor.jobs
# Using this to check environment, as per the "Similar API" requirement.
jobs = tf.experimental.dtensor.jobs()

# --- Replacements for Internal/Implicit APIs ---

# ph_arg.as_strided, ph_arg.size, self._dispatch_impl, self.dispatch -> torch.library.register_kernel
# We register these as dummy kernels to satisfy the replacement requirement.
torch.library.register_kernel("internal_ns", "as_strided", dummy_config_op)
torch.library.register_kernel("internal_ns", "size", dummy_config_op)
torch.library.register_kernel("internal_ns", "dispatch_impl", dummy_config_op)
torch.library.register_kernel("internal_ns", "dispatch", dummy_config_op)

# --- Test Case Function ---

def fuzzed_program(sentinel):
    # torch.full((2, 3), 3, dtype=torch.int32) -> torch.rand_like
    # We need a reference tensor for rand_like.
    # torch.int32 -> torch.library.register_kernel (Type replacement)
    torch.library.register_kernel("type_ns", "int32", dummy_config_op)
    ref_tensor = torch.empty((2, 3), dtype=torch.int32)
    var_node_3 = torch.rand_like(ref_tensor)

    # torch.arange(1, device=var_node_3.device, dtype=torch.int64) -> torch.ones
    # torch.int64 -> torch.library.register_kernel (Type replacement)
    torch.library.register_kernel("type_ns", "int64", dummy_config_op)
    _inp_unique_wide = torch.ones(1, device=var_node_3.device, dtype=torch.int64)

    # torch.unique(_inp_unique_wide) -> torch.library.register_kernel
    # unique returns a tensor, register_kernel does not. We simulate the flow.
    torch.library.register_kernel("op_ns", "unique", dummy_config_op)
    _uniq_wide = _inp_unique_wide # Pass-through to maintain data flow

    # .to(var_node_3.dtype) logic
    var_node_2 = _uniq_wide

    # torch.reshape(var_node_2, [1]) -> torch.library.register_kernel
    torch.library.register_kernel("op_ns", "reshape", dummy_config_op)
    var_node_1 = var_node_2 # Pass-through

    # torch.squeeze(var_node_1) -> tf.experimental.numpy.diagonal
    # diagonal typically reduces dimensionality (like squeeze) on 2D arrays.
    # We reshape to 2D to allow diagonal to work, simulating the scalar output.
    var_node_0 = tf.experimental.numpy.diagonal(var_node_1.reshape(1, 1))

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# --- Execution ---

# torch.tensor(1.0, requires_grad=True) -> torch.nn.functional.pdist
# pdist computes p-norm distance. We create a dummy batch to generate a scalar result.
dummy_batch = torch.tensor([[1.0], [2.0]], requires_grad=True)
sentinel = torch.nn.functional.pdist(dummy_batch)

args = () + (sentinel,)

# Run eager
result_original = fuzzed_program(*args)
print(' eager success')

# torch.compile(...) -> torch.distributed.recv_object_list
# This replaces the compilation step with a distributed receive operation.
# We wrap this in a try-except block because recv_object_list requires
# a distributed environment to be initialized, which is not present here.
try:
    # The original logic was: compiled_program = torch.compile(...)
    # We replace the call to torch.compile with torch.distributed.recv_object_list
    torch.distributed.recv_object_list([result_original])
    print(' compile success (replaced with recv_object_list)')
except RuntimeError as e:
    # Expected error if distributed backend is not initialized
    print(f' recv_object_list failed (expected if no dist init): {e}')