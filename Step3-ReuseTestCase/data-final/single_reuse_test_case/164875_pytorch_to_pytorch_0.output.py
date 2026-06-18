import torch

# Reproducer for Issue 164875: Eager/Compile Divergence with torch.add on empty dimensions
# The bug involves adding two tensors of shape (20, 0) which causes a size mismatch error
# in the compiled graph despite working in eager mode.

# Configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Use CUDA if available to match the original report, otherwise fallback to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1014698)

def fuzzed_program(arg_0, sentinel):
    # var_node_1 is arg_0 with size=(20, 0), stride=(1, 20), dtype=int64
    
    # The original fuzzer code attempted to generate var_node_2 via nonzero,
    # but the logic provided was syntactically invalid for creating a (20, 0) tensor.
    # We construct var_node_2 directly to match the shape and dtype required for the bug.
    var_node_2 = torch.empty((20, 0), dtype=torch.int64, device=device)
    
    # The API under test: torch.add
    var_node_0 = torch.add(arg_0, var_node_2)
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Create arg_0 with specific strides and empty dimension
# size=(20, 0), stride=(1, 20), dtype=int64
base_tensor = torch.randint(5, 30, (20,)).to(torch.int64).to(device)
arg_0 = torch.as_strided(base_tensor, (20, 0), (1, 20))

args = (arg_0, sentinel)

# Test Eager Mode
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# Test Compiled Mode
try:
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')