import torch
import torch.nn.functional as F

# Configuration from the bug report
# Guard against older PyTorch versions where _dynamo does not exist
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(52676)

# Determine device (CUDA is used in the original report, fallback to CPU for general runnability)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7):
    # Replicating the tensor operations from the original bug report
    var_node_6 = arg_0
    var_node_7 = arg_1
    var_node_5 = torch.matmul(var_node_6.to(torch.float64), var_node_7.to(torch.float64))
    var_node_9 = torch.full((9, 11, 12), 1.5758497316910556, dtype=torch.float64, device=device)
    var_node_10 = arg_2
    var_node_8 = torch.matmul(var_node_9.to(torch.float64), var_node_10.to(torch.float64))
    var_node_4 = torch.matmul(var_node_5.to(torch.float64), var_node_8.to(torch.float64))
    var_node_13 = arg_3
    var_node_14 = arg_4
    var_node_12 = torch.matmul(var_node_13.to(torch.float64), var_node_14.to(torch.float64))
    var_node_15 = arg_5
    var_node_11 = torch.matmul(var_node_12.to(torch.float64), var_node_15.to(torch.float64))
    var_node_3 = torch.matmul(var_node_4.to(torch.float64), var_node_11.to(torch.float64))
    var_node_17 = arg_6
    var_node_18 = arg_7
    var_node_16 = torch.matmul(var_node_17.to(torch.float64), var_node_18.to(torch.float64))
    var_node_2 = torch.matmul(var_node_3.to(torch.float64), var_node_16.to(torch.float64))
    var_node_23 = torch.full((156, 8), -0.5249394453404403, dtype=torch.float64, device=device)
    var_node_24 = torch.full((8, 9), 0.9331226188585692, dtype=torch.float64, device=device)
    var_node_22 = torch.matmul(var_node_23.to(torch.float64), var_node_24.to(torch.float64))
    
    # The original snippet cut off here, reconstructing based on pattern
    var_node_26 = torch.full((9, 13), -0.9276381954691514, dtype=torch.float64, device=device)

    # Replacing the original API (torch.nonzero) with torch.nn.functional.hardshrink
    # We apply hardshrink to the last generated tensor to verify behavior in this context
    return F.hardshrink(var_node_26)

# Generate inputs matching the shapes in the original report
inputs = [
    torch.randn(9, 9, 9, dtype=torch.float64, device=device),
    torch.randn(9, 9, 11, dtype=torch.float64, device=device),
    torch.randn(9, 12, 8, dtype=torch.float64, device=device),
    torch.randn(9, 8, 13, dtype=torch.float64, device=device),
    torch.randn(9, 13, 7, dtype=torch.float64, device=device),
    torch.randn(9, 7, 16, dtype=torch.float64, device=device),
    torch.randn(9, 16, 12, dtype=torch.float64, device=device),
    torch.randn(9, 12, 11, dtype=torch.float64, device=device),
]

# Run Eager
eager_output = fuzzed_program(*inputs)

# Run Compiled
# Check if torch.compile is available (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    compiled_program = torch.compile(fuzzed_program)
    compiled_output = compiled_program(*inputs)

    # Verify results match
    assert torch.allclose(eager_output, compiled_output), "Eager and Compiled outputs differ for hardshrink"
    print("Test passed.")
else:
    print("torch.compile is not available (requires PyTorch 2.0+). Skipping compiled execution check.")
    print("Eager execution passed.")