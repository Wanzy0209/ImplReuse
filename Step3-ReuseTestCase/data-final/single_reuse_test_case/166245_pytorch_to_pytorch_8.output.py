import torch
import torch._dynamo

# Setup device (CUDA is used in the original bug report, fallback to CPU for general runnability)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch.manual_seed(751735337)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, sentinel):
    # Reproducing the setup from the bug report
    var_node_4 = arg_0 # size=(15, 108, 4), dtype=int16
    var_node_3 = torch.chunk(var_node_4, 4, dim=1)[0] # size=(15, 27, 4)
    var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0] # size=(15, 27, 1)
    var_node_1 = torch.squeeze(var_node_2) # size=(15, 27)
    
    var_node_8 = torch.full((13, 27), 3, dtype=torch.int16, device=arg_0.device) # size=(13, 27)
    var_node_9 = arg_1 # size=(11,)
    
    _input_size_var_node_7 = var_node_8.size(0)
    _index_var_node_7 = torch.randint(0, _input_size_var_node_7, (11,), device=var_node_8.device)
    var_node_7 = torch.index_select(var_node_8, 0, _index_var_node_7) # size=(11, 27)
    var_node_6 = torch.clamp(var_node_7, min=-1.0, max=1.0) # size=(11, 27)

    # --- Adaptation: Replace/Insert torch.transpose here ---
    # The original report was about torch.gather. We test torch.transpose on the 
    # dynamically shaped tensor var_node_6.
    # Transpose dimensions 0 and 1: (11, 27) -> (27, 11)
    var_node_transposed = torch.transpose(var_node_6, 0, 1)
    
    # Continue with the rest of the program logic to ensure context is preserved
    var_node_12 = arg_2 # size=(3, 27)
    var_node_11 = torch.clamp(var_node_12, min=-1.0, max=1.0) # size=(3, 27)
    var_node_13 = torch.full((1,), 3, dtype=torch.int64, device=arg_0.device) # size=(1,)
    
    _input_size_var_node_10 = var_node_11.size(0)
    _index_var_node_10 = torch.randint(0, _input_size_var_node_10, (1,), device=var_node_11.device)
    var_node_10 = torch.index_select(var_node_11, 0, _index_var_node_10) # size=(1, 27)
    
    var_node_16 = arg_3 # size=(1, 27)
    var_node_17 = arg_4 # size=(1, 27)
    var_node_18 = arg_5 # size=(1, 27)
    var_node_15 = torch.cat([var_node_16, var_node_17, var_node_18], dim=0) # size=(3, 27)
    
    var_node_20 = arg_6 # size=(1,)
    var_node_19 = torch.clamp(var_node_20, min=None, max=1.0) # size=(1,)
    
    _input_size_var_node_14 = var_node_15.size(0)
    _index_var_node_14 = torch.randint(0, _input_size_var_node_14, (1,), device=var_node_15.device)
    var_node_14 = torch.index_select(var_node_15, 0, _index_var_node_14) # size=(1, 27)
    
    var_node_22 = torch.full((4, 27), 3, dtype=torch.int16, device=arg_0.device) # size=(4, 27)
    var_node_24 = torch.full((4,), 3, dtype=torch.int64, device=arg_0.device) # size=(4,)
    var_node_25 = torch.full((2,), 3, dtype=torch.int64, device=arg_0.device) # size=(2,)
    
    # Return the transposed tensor to verify the operation
    return var_node_transposed

# Generate inputs matching the shapes and dtypes from the bug report
arg_0 = torch.randint(0, 10, (15, 108, 4), dtype=torch.int16, device=device)
arg_1 = torch.randint(0, 10, (11,), dtype=torch.int64, device=device)
arg_2 = torch.randint(0, 10, (3, 27), dtype=torch.int16, device=device)
arg_3 = torch.randint(0, 10, (1, 27), dtype=torch.int16, device=device)
arg_4 = torch.randint(0, 10, (1, 27), dtype=torch.int16, device=device)
arg_5 = torch.randint(0, 10, (1, 27), dtype=torch.int16, device=device)
arg_6 = torch.randint(0, 10, (1,), dtype=torch.int64, device=device)

# Compile the function to trigger the Dynamo checks
try:
    compiled_f = torch.compile(fuzzed_program)
    result = compiled_f(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, None)
    
    # Verify output shape
    assert result.shape == (27, 11), f"Expected shape (27, 11), got {result.shape}"
    assert result.dtype == torch.int16, f"Expected dtype int16, got {result.dtype}"
    
    print("Test passed successfully.")
except Exception as e:
    print(f"Test failed with error: {e}")