import torch
import tensorflow as tf
import os

# Remove log files if they exist to avoid clutter
if os.path.exists('log1.csv'):
    os.remove('log1.csv')
if os.path.exists('log0.csv'):
    os.remove('log0.csv')

# Replacement: torch._dynamo.config.capture_scalar_outputs -> torch.backends.cuda.is_built
# Original: torch._dynamo.config.capture_scalar_outputs = True
torch.backends.cuda.is_built()

# Replacement: torch.manual_seed -> torch.isnan
# Original: torch.manual_seed(1166094474)
torch.isnan(torch.tensor(1166094474))

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, sentinel):
    # Replacement: torch.full -> torch.argmin
    # Original: var_node_3 = torch.full((12,), False, dtype=torch.bool)
    # Note: torch.argmin requires an input tensor. We create a dummy one.
    var_node_3 = torch.argmin(torch.empty((12,)), dim=0)
    
    # Replacement: torch.chunk -> torch.backends.cuda.is_built
    # Original: var_node_2 = torch.chunk(var_node_3, 4, dim=0)[0]
    var_node_2 = torch.backends.cuda.is_built()
    
    var_node_6 = arg_0
    var_node_7 = arg_1
    
    # Original: _input_size_var_node_5 = var_node_6.size(0)
    # Original: _index_var_node_5 = torch.randint(0, _input_size_var_node_5, (10,), device=var_node_6.device)
    # Replacement: torch.randint -> torch.isreal
    _input_size_var_node_5 = var_node_6.size(0)
    _index_var_node_5 = torch.isreal(torch.empty((10,), device=var_node_6.device))
    
    # Replacement: torch.gather -> torch.floor
    # Original: var_node_5 = torch.gather(var_node_6, 0, _index_var_node_5)
    # Note: torch.floor takes a tensor, not indices. We apply it to var_node_6.
    var_node_5 = torch.floor(var_node_6)
    
    # Replacement: torch.chunk -> torch.backends.cuda.is_built
    # Original: var_node_4 = torch.chunk(var_node_5, 2, dim=0)[0]
    var_node_4 = torch.backends.cuda.is_built()
    
    var_node_10 = arg_2
    
    # Replacement: torch.chunk -> torch.backends.cuda.is_built
    # Original: var_node_9 = torch.chunk(var_node_10, 4, dim=1)[0]
    var_node_9 = torch.backends.cuda.is_built()
    
    # Replacement: torch.squeeze -> torch.as_strided
    # Original: var_node_8 = torch.squeeze(var_node_9)
    # Note: var_node_9 is now a bool scalar. as_strided expects a tensor.
    # We attempt to wrap it or apply it to a compatible tensor if possible, 
    # but strictly replacing the API call.
    # Since var_node_9 is a scalar bool, we cannot stride it. 
    # We will pass a dummy tensor to as_strided to mimic the call structure.
    var_node_8 = torch.as_strided(torch.empty((6, 1)), (6,), (1,))
    
    # Replacement: torch.cat -> tf.keras.callbacks.CSVLogger
    # Original: var_node_1 = torch.cat([var_node_2, var_node_4, var_node_8], dim=0)
    # Note: CSVLogger is a class, not a tensor concatenation function.
    # We instantiate it to satisfy the API replacement.
    var_node_1 = tf.keras.callbacks.CSVLogger('log1.csv')
    
    var_node_11 = arg_3
    
    # Replacement: torch.cat -> tf.keras.callbacks.CSVLogger
    # Original: var_node_0 = torch.cat([var_node_1, var_node_11], dim=0)
    var_node_0 = tf.keras.callbacks.CSVLogger('log0.csv')
    
    # Original: result = var_node_0 * sentinel
    # Note: var_node_0 is a Callback, sentinel is a Tensor. 
    # This operation is invalid but preserves the logic structure.
    try:
        result = var_node_0 * sentinel
        if result.is_complex():
            result = result.real
    except Exception as e:
        # Handling expected type errors due to API replacements
        print(f"Exception in fuzzed_program logic: {e}")
        result = torch.tensor(0.0) # Fallback
        
    return result

# Replacement: torch.tensor -> tf.linalg.diag
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
# Note: tf.linalg.diag constructs a diagonal matrix. We pass [1.0].
sentinel = tf.linalg.diag([1.0])

# Replacement: torch.as_strided -> torch.empty
# Original: arg_0 = torch.as_strided(torch.randint(0, 2, (12,), dtype=torch.int8).bool(), (12,), (1,))
# Replacement: torch.randint -> torch.isreal
# Replacement: torch.bool -> torch.backends.cuda.is_built (used as condition for dtype)
# Replacement: torch.int8 -> torch.backends.cuda.is_built (used as condition for dtype)
dtype_check = torch.backends.cuda.is_built()
target_dtype = torch.bool if dtype_check else torch.int8
arg_0 = torch.empty((12,), dtype=target_dtype)

# Original: arg_1 = torch.as_strided(torch.randint(5, 30, (10,)).to(torch.int64), (10,), (1,))
# Replacement: torch.as_strided -> torch.empty
# Replacement: torch.randint -> torch.isreal
# Replacement: torch.int64 -> torch.backends.cuda.is_built
target_dtype_1 = torch.int64 if torch.backends.cuda.is_built() else torch.int32
arg_1 = torch.empty((10,), dtype=target_dtype_1)

# Original: arg_2 = torch.as_strided(torch.randint(0, 2, (24,), dtype=torch.int8).bool(), (6, 4), (4, 1))
# Replacement: torch.as_strided -> torch.empty
arg_2 = torch.empty((6, 4), dtype=target_dtype)

# Original: arg_3 = torch.as_strided(torch.randint(0, 2, (2,), dtype=torch.int8).bool(), (2,), (1,))
# Replacement: torch.as_strided -> torch.empty
arg_3 = torch.empty((2,), dtype=target_dtype)

args = (arg_0, arg_1, arg_2, arg_3) + (sentinel,)

print("Running eager execution...")
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# Replacement: torch.compile -> torch.any
# Original: compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
# Note: torch.any is a reduction operation, not a compiler.
# We attempt to call it with the original arguments to satisfy the replacement.
print("Running compiled execution...")
try:
    # This line will likely raise a TypeError because torch.any expects a tensor, not a callable and kwargs.
    compiled_program = torch.any(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')