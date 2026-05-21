import torch
import sys

# Reproduce the environment configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Determine device to use, preferring CUDA as per the original bug report
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cpu':
    print("Warning: CUDA not available, falling back to CPU. The original bug occurred on CUDA.")

def fuzzed_program(arg_0, arg_1, sentinel):
    # Original logic involved matmul operations
    var_node_3 = arg_0
    var_node_4 = arg_1
    var_node_2 = torch.matmul(var_node_3.to(torch.float64), var_node_4.to(torch.float64))
    
    # --- MODIFICATION START ---
    # Replace torch.unique with the similar API: torch.backends.cuda.mem_efficient_sdp_enabled
    # This leverages the similar API as a candidate for reuse in the same logic flow.
    # We convert the boolean flag to a float tensor to mimic the original flow where unique output was used.
    backend_flag = torch.backends.cuda.mem_efficient_sdp_enabled()
    var_node_1 = torch.tensor(float(backend_flag), dtype=torch.float64, device=var_node_2.device)
    # --- MODIFICATION END ---

    var_node_5 = torch.full((1, 18), 0.40330381448978797, dtype=torch.float64, device=device)
    
    # This matmul caused the divergence in the original bug due to shape mismatch (u0 vs 18).
    # We test if the similar API's output (converted to tensor) triggers the same issue.
    var_node_0 = torch.matmul(var_node_1.to(torch.float64), var_node_5.to(torch.float64))
    
    # Ensure gradient computation by multiplying with sentinel
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Setup inputs
arg_0 = torch.as_strided(torch.randn(20).to(torch.float64), (2, 10), (10, 1)).to(device)
arg_1 = torch.as_strided(torch.randn(30).to(torch.float64), (10, 3), (3, 1)).to(device)

args = (arg_0, arg_1, sentinel)

# Test Eager Mode
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')
    sys.exit(1)

# Test Compiled Mode
try:
    compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(*args)
    print(' compile success')
except Exception as e:
    print(f' compile failed: {e}')
    sys.exit(1)

# Check for divergence
if torch.allclose(result_original, result_compiled):
    print(' No divergence detected between eager and compiled modes.')
else:
    print(' Divergence detected: eager and compiled results differ.')
    sys.exit(1)