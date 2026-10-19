import torch
import torch.nn.functional as F
import torch.distributed
import tensorflow as tf

torch._dynamo.config.capture_scalar_outputs = True

# Replacement: torch.manual_seed -> tf.compat.v1.assert_variables_initialized
# Note: This TensorFlow function checks if variables are initialized. 
# It does not set a random seed like the original PyTorch function.
tf.compat.v1.assert_variables_initialized()

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0 # size=(17, 30, 17, 3), stride=(1530, 51, 3, 1), dtype=bool, device=cuda
    
    # Replacement: torch.chunk -> torch.as_strided
    # Original: torch.chunk(var_node_3, 3, dim=3)[0]
    # New: torch.as_strided(var_node_3, ...)[0]
    # Note: as_strided returns a Tensor, so [0] indexes the first element, 
    # whereas chunk returned a list of Tensors.
    var_node_2 = torch.as_strided(var_node_3, (17, 30, 17, 1), (510, 17, 1, 1))[0]
    
    # Replacement: torch.full -> torch.rand_like
    # Original: torch.full((17,), 3, dtype=torch.int64)
    # New: torch.rand_like(...)
    # Note: rand_like requires an input tensor to mimic shape and device.
    dummy_tensor = torch.empty(17, dtype=torch.int64, device=var_node_3.device)
    var_node_5 = torch.rand_like(dummy_tensor, dtype=torch.int64)
    
    var_node_6 = arg_1 # size=(15,), stride=(1,), dtype=int64, device=cuda
    _input_size_var_node_4 = var_node_5.size(0)
    
    # Replacement: torch.randint -> torch.imag
    # Original: torch.randint(0, _input_size_var_node_4, (15,), device=var_node_5.device)
    # New: torch.imag(...)
    # Note: imag extracts the imaginary part of a complex tensor.
    complex_input = torch.complex(torch.randn(15, device=var_node_5.device), torch.randn(15, device=var_node_5.device))
    _index_var_node_4 = torch.imag(complex_input)
    
    # Replacement: torch.gather -> torch.distributed.scatter
    # Original: var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4)
    # New: torch.distributed.scatter(...)
    # Note: scatter is a collective operation that modifies the tensor in place.
    # It requires a distributed process group to be initialized.
    var_node_4 = torch.empty_like(var_node_5)
    try:
        torch.distributed.scatter(var_node_4, [var_node_5], src=0)
    except RuntimeError:
        # Distributed operations fail without init_process_group.
        # Fallback to allow syntax checking, though logic is broken.
        pass

    _input_size_var_node_1 = var_node_2.size(0)
    
    # Replacement: torch.randint -> torch.imag
    complex_input_2 = torch.complex(torch.randn(15, device=var_node_2.device), torch.randn(15, device=var_node_2.device))
    _index_var_node_1 = torch.imag(complex_input_2)
    
    # Replacement: torch.index_select -> torch.empty_strided
    # Original: var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1)
    # New: torch.empty_strided(...)
    # Note: empty_strided creates an uninitialized tensor with given shape/stride.
    var_node_1 = torch.empty_strided((15, 30, 17, 1), (510, 17, 1, 1), device=var_node_2.device, dtype=var_node_2.dtype)
    
    # Original API Under Test: torch.squeeze
    var_node_0 = torch.squeeze(var_node_1) # size=(15, 30, 17), stride=(510, 17, 1), dtype=bool, device=cuda
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Replacement: torch.tensor -> torch.nn.functional.pdist
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# New: sentinel = torch.nn.functional.pdist(...)
# Note: pdist computes pairwise distances and returns a 1D tensor, not a scalar.
dummy_pdist_input = torch.randn(1, 2, requires_grad=True)
sentinel = F.pdist(dummy_pdist_input)

# Replacement: torch.as_strided -> torch.nn.functional.pdist
# Original: arg_0 = torch.as_strided(...)
# New: arg_0 = torch.nn.functional.pdist(...)
# Note: pdist returns a 1D tensor, but the program expects a 4D tensor.
arg_0 = F.pdist(torch.randn(10, 5))

arg_1 = torch.as_strided(torch.randint(5, 30, (15,)).to(torch.int64), (15,), (1,))

args = (arg_0, arg_1) + (sentinel,)

try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# Replacement: torch.compile -> torch.distributed.broadcast_object_list
# Original: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# New: torch.distributed.broadcast_object_list(...)
# Note: broadcast_object_list returns None, so compiled_program will be None.
# Calling None(*args) will raise a TypeError.
try:
    torch.distributed.broadcast_object_list([fuzzed_program], src=0)
    compiled_program = None # Result of broadcast_object_list is None
    result_compiled = compiled_program(*args)
    print(' compile success')
except TypeError as e:
    print(f' compile failed (expected due to replacement): {e}')
except RuntimeError as e:
    print(f' compile failed (distributed error): {e}')