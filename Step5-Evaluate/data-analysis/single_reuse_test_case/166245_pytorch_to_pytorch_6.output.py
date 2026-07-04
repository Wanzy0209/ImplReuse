import torch

# Check for torch._dynamo availability
try:
    import torch._dynamo
    # Configuration from the bug report
    torch._dynamo.config.capture_scalar_outputs = True
except (ImportError, ModuleNotFoundError):
    print("torch._dynamo not available, skipping test.")
    exit()

# Check for CUDA availability as the original bug report used device=cuda
if not torch.cuda.is_available():
    print("CUDA not available, skipping test.")
    exit()

torch.manual_seed(751735337)

# Recreate the inputs based on the comments in the bug report
# arg_0: size=(15, 108, 4), stride=(432, 1, 4), dtype=int16
# We use as_strided to match the specific non-contiguous stride pattern
arg_0 = torch.as_strided(
    torch.empty(15 * 108 * 4, dtype=torch.int16, device='cuda'),
    (15, 108, 4),
    (432, 1, 4)
)

arg_1 = torch.randint(0, 10, (11,), dtype=torch.int64, device='cuda')
arg_2 = torch.randint(0, 10, (3, 27), dtype=torch.int16, device='cuda')
arg_3 = torch.randint(0, 10, (1, 27), dtype=torch.int16, device='cuda')
arg_4 = torch.randint(0, 10, (1, 27), dtype=torch.int16, device='cuda')
arg_5 = torch.randint(0, 10, (1, 27), dtype=torch.int16, device='cuda')
arg_6 = torch.randint(0, 10, (1,), dtype=torch.int64, device='cuda')
sentinel = object()

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, sentinel):
    var_node_4 = arg_0
    var_node_3 = torch.chunk(var_node_4, 4, dim=1)[0]
    var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0]
    var_node_1 = torch.squeeze(var_node_2)
    var_node_8 = torch.full((13, 27), 3, dtype=torch.int16, device='cuda')
    var_node_9 = arg_1
    _input_size_var_node_7 = var_node_8.size(0)
    _index_var_node_7 = torch.randint(0, _input_size_var_node_7, (11,), device=var_node_8.device)
    var_node_7 = torch.index_select(var_node_8, 0, _index_var_node_7)
    var_node_6 = torch.clamp(var_node_7, min=-1.0, max=1.0)
    
    # --- Adaptation: Replace/Adapt original call site with torch.floor ---
    # The original API was torch.gather. We are testing torch.floor.
    # var_node_6 is int16. torch.floor on integers is a clone operation.
    # We apply it here to test the lowering and guard behavior.
    var_node_floor = torch.floor(var_node_6)
    
    # Continue with the rest of the program to ensure context is preserved
    var_node_12 = arg_2
    var_node_11 = torch.clamp(var_node_12, min=-1.0, max=1.0)
    var_node_13 = torch.full((1,), 3, dtype=torch.int64, device='cuda')
    _input_size_var_node_10 = var_node_11.size(0)
    _index_var_node_10 = torch.randint(0, _input_size_var_node_10, (1,), device=var_node_11.device)
    var_node_10 = torch.index_select(var_node_11, 0, _index_var_node_10)
    var_node_16 = arg_3
    var_node_17 = arg_4
    var_node_18 = arg_5
    var_node_15 = torch.cat([var_node_16, var_node_17, var_node_18], dim=0)
    var_node_20 = arg_6
    var_node_19 = torch.clamp(var_node_20, min=None, max=1.0)
    _input_size_var_node_14 = var_node_15.size(0)
    _index_var_node_14 = torch.randint(0, _input_size_var_node_14, (1,), device=var_node_15.device)
    var_node_14 = torch.index_select(var_node_15, 0, _index_var_node_14)
    var_node_22 = torch.full((4, 27), 3, dtype=torch.int16, device='cuda')
    var_node_24 = torch.full((4,), 3, dtype=torch.int64, device='cuda')
    var_node_25 = torch.full((2,), 3, dtype=torch.int64, device='cuda')
    _input_size_var_node_23 = var_node_24.size(0)
    
    return var_node_floor

# Run the test
print("Testing torch.floor with eager execution...")
eager_result = fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, sentinel)

print("Testing torch.floor with torch.compile...")
compiled_program = torch.compile(fuzzed_program)
compiled_result = compiled_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, sentinel)

# Verify results match
assert torch.equal(eager_result, compiled_result), "Eager and compiled results differ!"
print("Test passed.")