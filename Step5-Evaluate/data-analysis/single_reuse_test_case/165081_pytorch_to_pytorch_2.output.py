import torch
import torch.nn.functional as F

# Configuration from the bug report
# Added check for torch._dynamo availability to handle environments without it (e.g., PyTorch < 2.0)
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
else:
    print("Warning: torch._dynamo is not available. Skipping dynamo configuration.")

torch.manual_seed(52676)

# Check for CUDA availability as the bug report specifies device=cuda
if not torch.cuda.is_available():
    raise RuntimeError("Test requires CUDA to run as per the bug report context.")

device = torch.device("cuda")

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, sentinel):
    # Reconstructing the computation graph from the bug report
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
    
    # The original test case was truncated, but we adapt the logic here.
    # Original API: torch.nonzero (likely called on one of the nodes)
    # Adapted API: torch.nn.functional.softshrink
    
    # Applying softshrink to the result of the complex matmul chain
    return F.softshrink(var_node_2, lambd=0.5)

def generate_inputs():
    # Generating inputs based on the comments in the bug report
    # arg_0: size=(9, 9, 9), stride=(81, 9, 1), dtype=float64, device=cuda
    arg_0 = torch.randn(9, 9, 9, dtype=torch.float64, device=device)
    # arg_1: size=(9, 9, 11), stride=(99, 11, 1), dtype=float64, device=cuda
    arg_1 = torch.randn(9, 9, 11, dtype=torch.float64, device=device)
    # arg_2: size=(9, 12, 8), stride=(96, 8, 1), dtype=float64, device=cuda
    arg_2 = torch.randn(9, 12, 8, dtype=torch.float64, device=device)
    # arg_3: size=(9, 8, 13), stride=(104, 13, 1), dtype=float64, device=cuda
    arg_3 = torch.randn(9, 8, 13, dtype=torch.float64, device=device)
    # arg_4: size=(9, 13, 7), stride=(91, 7, 1), dtype=float64, device=cuda
    arg_4 = torch.randn(9, 13, 7, dtype=torch.float64, device=device)
    # arg_5: size=(9, 7, 16), stride=(112, 16, 1), dtype=float64, device=cuda
    arg_5 = torch.randn(9, 7, 16, dtype=torch.float64, device=device)
    # arg_6: size=(9, 16, 12), stride=(192, 12, 1), dtype=float64, device=cuda
    arg_6 = torch.randn(9, 16, 12, dtype=torch.float64, device=device)
    # arg_7: size=(9, 12, 11), stride=(132, 11, 1), dtype=float64, device=cuda
    arg_7 = torch.randn(9, 12, 11, dtype=torch.float64, device=device)
    
    # Remaining args (8-18) are not explicitly defined in the snippet but are required by the function signature
    # We generate dummy tensors for them
    args = [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7]
    for _ in range(11):
        args.append(torch.randn(10, 10, dtype=torch.float64, device=device))
    
    args.append(None) # sentinel
    return args

# Test execution
inputs = generate_inputs()

# 1. Eager execution
eager_result = fuzzed_program(*inputs)

# 2. Compiled execution
# Check if torch.compile is available (requires PyTorch 2.0+)
if hasattr(torch, 'compile'):
    compiled_program = torch.compile(fuzzed_program)
    compiled_result = compiled_program(*inputs)

    # 3. Assertion
    # Check if eager and compiled results match
    assert torch.allclose(eager_result, compiled_result), "Eager and compiled results diverge!"
    print("Test passed successfully.")
else:
    print("torch.compile is not available in this environment. Skipping compiled execution and assertion.")