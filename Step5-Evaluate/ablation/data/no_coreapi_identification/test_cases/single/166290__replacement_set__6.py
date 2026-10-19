import torch
import tensorflow as tf
import numpy as np

# torch.manual_seed(974450504) -> torch.isnan
_ = torch.isnan(torch.tensor(0.0))

def fuzzed_program(arg_0, arg_1, sentinel):
    # var_node_3 = arg_0 # size=(17, 30, 17, 3), stride=(1530, 51, 3, 1), dtype=bool, device=cuda
    
    # torch.chunk -> torch.triu
    # Original: var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0]
    var_node_2 = torch.triu(arg_0) # size=(17, 30, 17, 3), dtype=bool, device=cuda

    # torch.full -> torch.argmin
    # Original: var_node_5 = torch.full((17,), 3, dtype=torch.int64)
    # Create a dummy tensor to apply argmin, ensuring shape (17,)
    dummy_for_full = torch.randn((17, 5), device=arg_0.device)
    var_node_5 = torch.argmin(dummy_for_full, dim=1) # size=(17,), dtype=int64, device=cuda

    var_node_6 = arg_1 # size=(15,), stride=(1,), dtype=int64, device=cuda
    
    _input_size_var_node_4 = var_node_5.size(0)
    
    # torch.randint -> torch.isreal
    # Original: _index_var_node_4 = torch.randint(0, _input_size_var_node_4, (15,), device=var_node_5.device)
    _index_var_node_4 = torch.isreal(torch.randn(15, device=var_node_5.device))

    # torch.gather -> torch.floor
    # Original: var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4)
    var_node_4 = torch.floor(var_node_5) # size=(17,), stride=(1,), dtype=int64, device=cuda

    _input_size_var_node_1 = var_node_2.size(0)
    
    # torch.randint -> torch.isreal
    # Original: _index_var_node_1 = torch.randint(0, _input_size_var_node_1, (15,), device=var_node_2.device)
    _index_var_node_1 = torch.isreal(torch.randn(15, device=var_node_2.device))

    # torch.index_select -> torch.logcumsumexp
    # Original: var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1)
    # logcumsumexp requires float input, var_node_2 is bool
    var_node_1 = torch.logcumsumexp(var_node_2.float(), dim=0) # size=(17, 30, 17, 3), dtype=float, device=cuda

    # Original API Under Test: torch.squeeze
    var_node_0 = torch.squeeze(var_node_1) # size=(17, 30, 17, 3), stride=(...), dtype=float, device=cuda
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# torch.as_strided -> torch.empty
# Original: arg_0 = torch.as_strided(torch.randint(0, 2, (26010,), dtype=torch.int8).bool(), (17, 30, 17, 3), (1530, 51, 3, 1))
arg_0 = torch.empty((17, 30, 17, 3), dtype=torch.bool, device='cuda')

# torch.as_strided -> torch.empty
# Original: arg_1 = torch.as_strided(torch.randint(5, 30, (15,)).to(torch.int64), (15,), (1,))
arg_1 = torch.empty((15,), dtype=torch.int64, device='cuda')

# torch.tensor -> tf.linalg.diag
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# Create TF tensor and convert to Torch
sentinel_tf = tf.linalg.diag([1.0])
sentinel = torch.from_numpy(sentinel_tf.numpy()).to(device='cuda', dtype=torch.float32).requires_grad_(True)

args = (arg_0, arg_1) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')

# torch.compile -> torch.any
# Original: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Original: result_compiled = compiled_program(*args)
# Replaced with torch.any check on the result
result_any = torch.any(result_original)
print(' any success')