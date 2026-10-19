import torch
import tensorflow as tf
import sys

# Configuration
torch._dynamo.config.capture_scalar_outputs = True

# Replacement: torch.manual_seed -> torch.rand_like
# Original: torch.manual_seed(974450504)
# Replacement: torch.rand_like requires an input tensor. We create a dummy one to satisfy the call.
_ = torch.rand_like(torch.tensor(0.0))

def fuzzed_program(arg_0, arg_1, sentinel):
    try:
        var_node_3 = arg_0 # size=(17, 30, 17, 3), stride=(1530, 51, 3, 1), dtype=bool, device=cuda
        
        # Replacement: torch.chunk -> torch.empty
        # Original: var_node_2 = torch.chunk(var_node_3, 3, dim=3)[0]
        # Replacement: torch.empty creates an uninitialized tensor with the specified size.
        var_node_2 = torch.empty((17, 30, 17, 1), dtype=torch.bool, device='cuda')
        
        # Replacement: torch.full -> torch.is_complex
        # Original: var_node_5 = torch.full((17,), 3, dtype=torch.int64)
        # Replacement: torch.is_complex checks if the tensor is complex. Returns a bool.
        # This changes the type from Tensor to bool, which will likely break subsequent logic.
        var_node_5 = torch.is_complex(torch.empty((17,), dtype=torch.int64, device='cuda'))
        
        # The following line will fail because var_node_5 is a bool, not a tensor.
        # We wrap the rest in a try-except to handle the type mismatch gracefully for the test case.
        _input_size_var_node_4 = var_node_5.size(0)
        
        # Replacement: torch.randint -> torch.bernoulli
        # Original: _index_var_node_4 = torch.randint(0, _input_size_var_node_4, (15,), device=var_node_5.device)
        # Replacement: torch.bernoulli generates 0/1. We cast to long to mimic integer indices.
        _index_var_node_4 = torch.bernoulli(torch.full((15,), 0.5, device='cuda')).long()
        
        # Replacement: torch.gather -> torch.scatter_add
        # Original: var_node_4 = torch.gather(var_node_5, 0, _index_var_node_4)
        # Replacement: torch.scatter_add is the inverse of gather. We adapt the call.
        # Note: var_node_5 is a bool here, which might still cause issues, but we follow the replacement.
        var_node_4 = torch.scatter_add(torch.zeros((15,), dtype=torch.int64, device='cuda'), 0, _index_var_node_4, var_node_5)
        
        _input_size_var_node_1 = var_node_2.size(0)
        
        # Replacement: torch.randint -> torch.bernoulli
        _index_var_node_1 = torch.bernoulli(torch.full((15,), 0.5, device='cuda')).long()
        
        # Replacement: torch.index_select -> tf.keras.ops.triu
        # Original: var_node_1 = torch.index_select(var_node_2, 0, _index_var_node_1)
        # Replacement: tf.keras.ops.triu returns the upper triangular part of a tensor.
        # We convert the torch tensor to a tensorflow tensor.
        var_node_1 = tf.keras.ops.triu(tf.convert_to_tensor(var_node_2))
        
        # Original API Under Test: torch.squeeze
        # Note: var_node_1 is now a TensorFlow tensor. torch.squeeze expects a torch tensor.
        # This will raise a TypeError. We catch it below.
        var_node_0 = torch.squeeze(var_node_1)
        
        # Ensure gradient computation by multiplying with sentinel and taking real part
        result = var_node_0 * sentinel
        if result.is_complex():
            result = result.real
        return result
    except Exception as e:
        print(f" Runtime error in fuzzed_program due to API replacements: {type(e).__name__}: {e}")
        # Return a dummy tensor to allow the script to continue if needed, or re-raise
        return torch.tensor(0.0)

# Sentinel tensor to ensure gradient computation
# Replacement: torch.tensor -> torch.empty_strided
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# Replacement: torch.empty_strided creates an uninitialized tensor.
sentinel = torch.empty_strided((1,), (1,), dtype=torch.float32, requires_grad=True)

# Replacement: torch.as_strided -> torch.chunk
# Original: arg_0 = torch.as_strided(torch.randint(0, 2, (26010,), dtype=torch.int8).bool(), (17, 30, 17, 3), (1530, 51, 3, 1))
# Replacement: torch.chunk splits a tensor into chunks. We use chunks=1 to keep the original size (17, 30, 17, 3).
arg_0 = torch.chunk(torch.randint(0, 2, (17, 30, 17, 3), dtype=torch.int8).bool(), 1, dim=3)[0]

# Replacement: torch.as_strided -> torch.chunk
# Original: arg_1 = torch.as_strided(torch.randint(5, 30, (15,)).to(torch.int64), (15,), (1,))
# Replacement: torch.chunk with chunks=1.
arg_1 = torch.chunk(torch.randint(5, 30, (15,)).to(torch.int64), 1, dim=0)[0]

args = (arg_0, arg_1) + (sentinel,)

print("Running eager execution...")
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

print("Running compiled execution...")
# Replacement: torch.compile -> torch.distributed.recv_object_list
# Original: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Replacement: torch.distributed.recv_object_list is a distributed communication primitive and does not return a compiled function.
# We assign the result of the replacement, but we cannot call it as a function.
try:
    # This line is syntactically valid but semantically incorrect for the subsequent call.
    # We wrap it to avoid crashing the script on the assignment itself if init is needed, 
    # though recv_object_list usually requires an initialized process group.
    compiled_program = torch.distributed.recv_object_list([None]) 
except Exception as e:
    print(f" Could not initialize replacement for torch.compile: {e}")
    compiled_program = None

if compiled_program is not None and callable(compiled_program):
    try:
        result_compiled = compiled_program(*args)
        print(' compile success')
    except Exception as e:
        print(f' compile failed: {e}')
else:
    print(' Skipping compiled execution: replacement API is not a callable function.')