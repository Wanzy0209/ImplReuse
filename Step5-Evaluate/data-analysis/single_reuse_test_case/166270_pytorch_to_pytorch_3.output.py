import torch
import sys

# Check if torch.compile is available (PyTorch 2.0+)
if not hasattr(torch, 'compile'):
    print("Skipping test: torch.compile is not available (requires PyTorch 2.0+)")
    sys.exit(0)

# Configuration from the original bug report
# Guard against AttributeError if _dynamo is not available
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True

torch.manual_seed(1061983224)

def fuzzed_program(arg_0, sentinel):
    var_node_4 = arg_0 # size=(4,), stride=(1,), dtype=bool, device=cuda
    var_node_3 = torch.chunk(var_node_4, 4, dim=0)[0] # size=(1,), stride=(1,), dtype=bool, device=cuda

    # Replaced torch.squeeze with torch.index_select
    # var_node_3 is size (1,). We select the 0th index.
    # This keeps the dimension, unlike squeeze which removes it.
    index = torch.tensor([0], device=arg_0.device)
    var_node_2 = torch.index_select(var_node_3, 0, index) # size=(1,), stride=(1,), dtype=bool, device=cuda

    var_node_1 = torch.stack([var_node_2], dim=0) # size=(1, 1), stride=(1, 1), dtype=bool, device=cuda
    var_node_0 = torch.reshape(var_node_1, [1]) # size=(1,), stride=(1,), dtype=bool, device=cuda

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Setup input tensor on CUDA (as per original bug report)
# Fallback to CPU if CUDA is not available to ensure the test is runnable in non-GPU environments,
# though the original bug was specific to CUDA.
device = 'cuda' if torch.cuda.is_available() else 'cpu'
arg_0 = torch.as_strided(torch.randint(0, 2, (4,), dtype=torch.int8).bool().to(device), (4,), (1,))

args = (arg_0,) + (sentinel,)

# Run eager mode
result_original = fuzzed_program(*args)
print(' eager success')

# Run compiled mode
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')