import torch

# Configuration from the original bug report
# Check if torch._dynamo exists (PyTorch 2.0+)
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
else:
    print("Warning: torch._dynamo not found. Skipping dynamo configuration. PyTorch 2.0+ required for torch.compile.")

torch.manual_seed(1012969)

def fuzzed_program(arg_0, arg_1, sentinel):
    var_node_3 = arg_0 # size=(2, 10), stride=(10, 1), dtype=float64, device=cuda
    var_node_4 = arg_1 # size=(10, 3), stride=(3, 1), dtype=float64, device=cuda
    var_node_2 = torch.matmul(var_node_3.to(torch.float64), var_node_4.to(torch.float64)) # size=(2, 3), stride=(3, 1), dtype=float64, device=cuda
    
    # --- Adaptation Start ---
    # Original: _inp_unique_wide = torch.arange(1, device=var_node_2.device, dtype=torch.int64)
    # Original: _uniq_wide = torch.unique(_inp_unique_wide)
    # Original: var_node_1 = _uniq_wide.to(var_node_2.dtype)
    
    # Replaced torch.unique with torch.numel
    _inp_numel = torch.arange(1, device=var_node_2.device, dtype=torch.int64)
    numel_val = torch.numel(_inp_numel)
    # torch.numel returns an int, so we wrap it back in a tensor to maintain compatibility 
    # with the subsequent matmul operation, mimicking the original tensor flow.
    var_node_1 = torch.tensor(numel_val, dtype=var_node_2.dtype, device=var_node_2.device)
    # --- Adaptation End ---

    var_node_5 = torch.full((1, 18), 0.40330381448978797, dtype=torch.float64) # size=(1, 18), stride=(18, 1), dtype=float64, device=cuda
    var_node_0 = torch.matmul(var_node_1.to(torch.float64), var_node_5.to(torch.float64)) # size=(18,), stride=(1,), dtype=float64, device=cuda
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.as_strided(torch.randn(20).to(torch.float64), (2, 10), (10, 1))
arg_1 = torch.as_strided(torch.randn(30).to(torch.float64), (10, 3), (3, 1))

args = (arg_0, arg_1, sentinel)

# Test Eager Mode
try:
    result_original = fuzzed_program(*args)
    print(' eager success')
except Exception as e:
    print(f' eager failed: {e}')

# Test Compiled Mode
if not hasattr(torch, 'compile'):
    print(' compile skipped: torch.compile not available (requires PyTorch 2.0+)')
else:
    try:
        compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
        result_compiled = compiled_program(*args)
        print(' compile success')
        
        # Verify results match (allowing for minor floating point differences)
        assert torch.allclose(result_original, result_compiled, atol=1e-6), "Divergence detected between eager and compiled results!"
        print(' results match')
    except Exception as e:
        print(f' compile failed: {e}')