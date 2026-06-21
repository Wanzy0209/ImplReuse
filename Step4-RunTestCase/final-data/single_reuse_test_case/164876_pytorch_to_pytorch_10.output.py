import torch

# Setup configurations from the original bug report
# Check if _dynamo is available to prevent AttributeError in environments without it
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(1012969)

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0 # size=(2, 10), stride=(10, 1), dtype=float64
    var_node_4 = arg_1 # size=(10, 3), stride=(3, 1), dtype=float64
    var_node_2 = torch.matmul(var_node_3.to(torch.float64), var_node_4.to(torch.float64)) # size=(2, 3), stride=(3, 1), dtype=float64

    # --- Adapted Section ---
    # Original API: torch.unique
    # Similar API: torch.triu_indices
    # torch.triu_indices returns a tuple of tensors. We select the first one to maintain the flow.
    # Using dimensions (1, 1) to produce a tensor of size (1,), similar to the original unique output.
    _triu_res = torch.triu_indices(1, 1, device=var_node_2.device)
    var_node_1 = _triu_res[0].to(var_node_2.dtype) # size=(1,), stride=(1,), dtype=float64
    # -----------------------

    var_node_5 = torch.full((1, 18), 0.40330381448978797, dtype=torch.float64) # size=(1, 18), stride=(18, 1), dtype=float64
    var_node_0 = torch.matmul(var_node_1.to(torch.float64), var_node_5.to(torch.float64)) # size=(18,), stride=(1,), dtype=float64

    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Determine device (CUDA or CPU) to ensure runnability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arg_0 = torch.as_strided(torch.randn(20, device=device).to(torch.float64), (2, 10), (10, 1))
arg_1 = torch.as_strided(torch.randn(30, device=device).to(torch.float64), (10, 3), (3, 1))

args = (arg_0, arg_1, sentinel)

# Run Eager
result_original = fuzzed_program(*args)
print(' eager success')

# Run Compiled
# Check if torch.compile is available
if hasattr(torch, 'compile'):
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')

    # Verify results match
    assert torch.allclose(result_original, result_compiled), "Eager and compiled results differ"
else:
    print(" torch.compile not available, skipping compiled test")