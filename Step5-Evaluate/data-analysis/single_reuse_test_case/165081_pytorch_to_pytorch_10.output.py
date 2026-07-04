import torch

# Configuration from the bug report
# Fix: Check if _dynamo exists before accessing its config to handle older PyTorch versions
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(52676)

def fuzzed_program():
    # Reconstructing inputs based on comments in the original snippet
    # Using torch.randn to generate the required tensors on CUDA with float64
    arg_0 = torch.randn(9, 9, 9, dtype=torch.float64, device='cuda')
    arg_1 = torch.randn(9, 9, 11, dtype=torch.float64, device='cuda')
    arg_2 = torch.randn(9, 12, 8, dtype=torch.float64, device='cuda')
    arg_3 = torch.randn(9, 8, 13, dtype=torch.float64, device='cuda')
    arg_4 = torch.randn(9, 13, 7, dtype=torch.float64, device='cuda')
    arg_5 = torch.randn(9, 7, 16, dtype=torch.float64, device='cuda')
    arg_6 = torch.randn(9, 16, 12, dtype=torch.float64, device='cuda')
    arg_7 = torch.randn(9, 12, 11, dtype=torch.float64, device='cuda')

    var_node_6 = arg_0
    var_node_7 = arg_1
    var_node_5 = torch.matmul(var_node_6.to(torch.float64), var_node_7.to(torch.float64))
    
    var_node_9 = torch.full((9, 11, 12), 1.5758497316910556, dtype=torch.float64, device='cuda')
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

    # Original API: torch.nonzero
    # Similar API: torch.argmin
    # Replacing the call site with torch.argmin
    return torch.argmin(var_node_2)

# Compile and run the test
try:
    # Fix: Check if torch.compile exists, otherwise run eagerly
    if hasattr(torch, 'compile'):
        compiled_fn = torch.compile(fuzzed_program)
        result = compiled_fn()
    else:
        print("torch.compile not available, running in eager mode.")
        result = fuzzed_program()
    print("Test passed. Result:", result)
except Exception as e:
    print(f"Test failed with error: {e}")