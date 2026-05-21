import torch
import torch.nn.functional as F

# Reproduce the configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Set the seed to ensure reproducibility of the fuzzer inputs
torch.manual_seed(52676)

def get_inputs():
    """
    Generates the input tensors based on the comments found in the 
    original bug report's fuzzed_program.
    """
    # Use CUDA if available, otherwise CPU (to ensure test runs everywhere)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Shapes inferred from the original bug report comments
    # arg_0: size=(9, 9, 9), dtype=float64
    # arg_1: size=(9, 9, 11), dtype=float64
    # arg_2: size=(9, 12, 8), dtype=float64
    # arg_3: size=(9, 8, 13), dtype=float64
    # arg_4: size=(9, 13, 7), dtype=float64
    # arg_5: size=(9, 7, 16), dtype=float64
    # arg_6: size=(9, 16, 12), dtype=float64
    # arg_7: size=(9, 12, 11), dtype=float64
    shapes = [
        (9, 9, 9), (9, 9, 11), (9, 12, 8), (9, 8, 13),
        (9, 13, 7), (9, 7, 16), (9, 16, 12), (9, 12, 11)
    ]
    
    inputs = []
    for shape in shapes:
        inputs.append(torch.randn(shape, dtype=torch.float64, device=device))
    return inputs, device

def fuzzed_program_with_tanh(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, device):
    """
    Reconstructs the logic from the original bug report but applies 
    torch.nn.functional.tanh (the similar API) to the result.
    """
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
    
    # Apply the similar API: torch.nn.functional.tanh
    # This tests if tanh behaves correctly in the graph context where the original bug occurred.
    return F.tanh(var_node_22)

def test_tanh_compile_divergence():
    inputs, device = get_inputs()
    
    # 1. Run in Eager mode
    eager_result = fuzzed_program_with_tanh(*inputs, device=device)
    
    # 2. Run in Compiled mode (torch._dynamo)
    # The original bug was an Eager/Compile divergence.
    compiled_fn = torch.compile(fuzzed_program_with_tanh)
    compiled_result = compiled_fn(*inputs, device=device)
    
    # 3. Assert results are close
    # If the bug "Could not guard on data-dependent expression" affects this graph with tanh,
    # the compilation might fail or produce incorrect results.
    assert torch.allclose(eager_result, compiled_result, atol=1e-5), \
        f"Divergence detected between eager and compiled results.\nEager:\n{eager_result}\nCompiled:\n{compiled_result}"

if __name__ == "__main__":
    test_tanh_compile_divergence()
    print("Test passed successfully.")