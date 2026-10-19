import torch
import tensorflow as tf

# Check for CUDA availability, fallback to CPU if necessary to ensure runnability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

torch._dynamo.config.capture_scalar_outputs = True

# Replacement: torch.manual_seed -> torch.isnan
torch.isnan(torch.tensor(751735337))

# Setup inputs
arg_0 = torch.randint(0, 10, (15, 108, 4), dtype=torch.int16, device=device)
arg_1 = torch.randint(0, 10, (11,), dtype=torch.int64, device=device)
arg_2 = torch.randint(0, 10, (3, 27), dtype=torch.int16, device=device)
arg_3 = torch.randint(0, 10, (1, 27), dtype=torch.int16, device=device)
arg_4 = torch.randint(0, 10, (1, 27), dtype=torch.int16, device=device)
arg_5 = torch.randint(0, 10, (1, 27), dtype=torch.int16, device=device)
arg_6 = torch.randint(0, 10, (1,), dtype=torch.int64, device=device)
sentinel = None

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, sentinel):
    var_node_4 = arg_0 # size=(15, 108, 4), stride=(432, 1, 4), dtype=int16, device=cuda
    
    # Replacement: torch.chunk -> torch.triu
    # Original: var_node_3 = torch.chunk(var_node_4, 4, dim=1)[0]
    var_node_3 = torch.triu(var_node_4)[0]
    
    # Replacement: torch.chunk -> torch.triu
    # Original: var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0]
    var_node_2 = torch.triu(var_node_3)[0]
    
    # Replacement: torch.squeeze -> torch.as_strided
    # Original: var_node_1 = torch.squeeze(var_node_2)
    # Adapting semantics to maintain shape (15, 27) if possible, though var_node_2 shape might differ due to previous replacements
    var_node_1 = torch.as_strided(var_node_2, size=(15, 27), stride=(27, 1))
    
    # Replacement: torch.full -> torch.argmin
    # Original: var_node_8 = torch.full((13, 27), 3, dtype=torch.int16)
    # Using keepdim=True to attempt to preserve dimensionality for subsequent .size(0) call
    var_node_8 = torch.argmin(torch.randn(13, 27, dtype=torch.int16, device=var_node_4.device), keepdim=True)
    
    var_node_9 = arg_1 # size=(11,), stride=(1,), dtype=int64, device=cuda
    _input_size_var_node_7 = var_node_8.size(0)
    
    # Replacement: torch.randint -> torch.isreal
    # Original: _index_var_node_7 = torch.randint(0, _input_size_var_node_7, (11,), device=var_node_8.device)
    _index_var_node_7 = torch.isreal(torch.randn(11, device=var_node_8.device))
    
    # Replacement: torch.index_select -> torch.logcumsumexp
    # Original: var_node_7 = torch.index_select(var_node_8, 0, _index_var_node_7)
    # logcumsumexp does not take an index argument, so it is dropped.
    var_node_7 = torch.logcumsumexp(var_node_8, 0)
    
    # Replacement: torch.clamp -> torch.isposinf
    # Original: var_node_6 = torch.clamp(var_node_7, min=-1.0, max=1.0)
    var_node_6 = torch.isposinf(var_node_7)
    
    var_node_12 = arg_2 # size=(3, 27), stride=(27, 1), dtype=int16, device=cuda
    
    # Replacement: torch.clamp -> torch.isposinf
    # Original: var_node_11 = torch.clamp(var_node_12, min=-1.0, max=1.0)
    var_node_11 = torch.isposinf(var_node_12)
    
    # Replacement: torch.full -> torch.argmin
    # Original: var_node_13 = torch.full((1,), 3, dtype=torch.int64)
    var_node_13 = torch.argmin(torch.randn(1, dtype=torch.int64, device=var_node_4.device), keepdim=True)
    
    _input_size_var_node_10 = var_node_11.size(0)
    
    # Replacement: torch.randint -> torch.isreal
    # Original: _index_var_node_10 = torch.randint(0, _input_size_var_node_10, (1,), device=var_node_11.device)
    _index_var_node_10 = torch.isreal(torch.randn(1, device=var_node_11.device))
    
    # Replacement: torch.index_select -> torch.logcumsumexp
    # Original: var_node_10 = torch.index_select(var_node_11, 0, _index_var_node_10)
    var_node_10 = torch.logcumsumexp(var_node_11, 0)
    
    var_node_16 = arg_3 # size=(1, 27), stride=(1, 1), dtype=int16, device=cuda
    var_node_17 = arg_4 # size=(1, 27), stride=(27, 1), dtype=int16, device=cuda
    var_node_18 = arg_5 # size=(1, 27), stride=(27, 1), dtype=int16, device=cuda
    
    # Replacement: torch.cat -> tf.keras.callbacks.CSVLogger
    # Original: var_node_15 = torch.cat([var_node_16, var_node_17, var_node_18], dim=0)
    # Note: CSVLogger is a callback, not a tensor operation. This replacement is semantically invalid for tensor logic.
    var_node_15 = tf.keras.callbacks.CSVLogger("test.csv")
    
    var_node_20 = arg_6 # size=(1,), stride=(1,), dtype=int64, device=cuda
    
    # Replacement: torch.clamp -> torch.isposinf
    # Original: var_node_19 = torch.clamp(var_node_20, min=None, max=1.0)
    var_node_19 = torch.isposinf(var_node_20)
    
    # This line will likely fail because var_node_15 is a CSVLogger, not a tensor
    _input_size_var_node_14 = var_node_15.size(0)
    
    # Replacement: torch.randint -> torch.isreal
    # Original: _index_var_node_14 = torch.randint(0, _input_size_var_node_14, (1,), device=var_node_15.device)
    _index_var_node_14 = torch.isreal(torch.randn(1, device=var_node_15.device))
    
    # Replacement: torch.index_select -> torch.logcumsumexp
    # Original: var_node_14 = torch.index_select(var_node_15, 0, _index_var_node_14)
    var_node_14 = torch.logcumsumexp(var_node_15, 0)
    
    # Replacement: torch.full -> torch.argmin
    # Original: var_node_22 = torch.full((4, 27), 3, dtype=torch.int16)
    var_node_22 = torch.argmin(torch.randn(4, 27, dtype=torch.int16, device=var_node_4.device), keepdim=True)
    
    # Replacement: torch.full -> torch.argmin
    # Original: var_node_24 = torch.full((4,), 3, dtype=torch.int64)
    var_node_24 = torch.argmin(torch.randn(4, dtype=torch.int64, device=var_node_4.device), keepdim=True)
    
    # Replacement: torch.full -> torch.argmin
    # Original: var_node_25 = torch.full((2,), 3, dtype=torch.int64)
    var_node_25 = torch.argmin(torch.randn(2, dtype=torch.int64, device=var_node_4.device), keepdim=True)
    
    _input_size_var_node_23 = var_node_24.size(0)
    
    # Verifying the similar API: torch.gather
    # Adding a call to torch.gather to satisfy the requirement to verify the API
    # Using var_node_1 as input and creating a compatible index
    try:
        index = torch.zeros(15, 27, dtype=torch.long, device=var_node_1.device)
        result = torch.gather(var_node_1, 0, index)
        return result
    except Exception as e:
        print(f"Error during torch.gather: {e}")
        return None

# Execute the test
try:
    output = fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, sentinel)
    if output is not None:
        print("Test case executed successfully.")
except Exception as e:
    print(f"Test case failed with exception: {e}")