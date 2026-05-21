import torch

# Configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch.manual_seed(52676)

def fuzzed_program(arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, arg_13, arg_14, arg_15, arg_16, arg_17, arg_18, sentinel):
    # Setup logic from the original test case
    var_node_6 = arg_0
    var_node_7 = arg_1
    var_node_5 = torch.matmul(var_node_6.to(torch.float64), var_node_7.to(torch.float64))
    var_node_9 = torch.full((9, 11, 12), 1.5758497316910556, dtype=torch.float64)
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
    var_node_23 = torch.full((156, 8), -0.5249394453404403, dtype=torch.float64)
    var_node_24 = torch.full((8, 9), 0.9331226188585692, dtype=torch.float64)
    var_node_22 = torch.matmul(var_node_23.to(torch.float64), var_node_24.to(torch.float64))
    var_node_26 = torch.full((9, 13), -0.9276381954691514, dtype=torch.float64)

    # Adaptation: Replace torch.nonzero with torch.floor_divide
    # We use var_node_2 and var_node_5 as they have compatible shapes (9, 9, 11)
    return torch.floor_divide(var_node_2, var_node_5)

# Prepare inputs
# Using CPU to ensure the test is runnable on any machine, though the bug report used CUDA.
# Shapes and dtypes are preserved based on the comments in the bug report.
inputs = []
# Explicit shapes from comments
shapes = [
    (9, 9, 9), (9, 9, 11), (9, 12, 8), (9, 8, 13), (9, 13, 7),
    (9, 7, 16), (9, 16, 12), (9, 12, 11)
]
# Remaining args (arg_8 to arg_18) are not explicitly defined in the snippet, 
# so we generate dummy tensors to satisfy the function signature.
for _ in range(19):
    if _ < len(shapes):
        inputs.append(torch.randn(shapes[_], dtype=torch.float64))
    else:
        # Generate arbitrary shapes for unused args
        inputs.append(torch.randn(10, 10, dtype=torch.float64))
inputs.append(None) # sentinel

# Test Eager
eager_result = fuzzed_program(*inputs)

# Test Compiled
compiled_program = torch.compile(fuzzed_program)
compiled_result = compiled_program(*inputs)

# Assertion
assert torch.allclose(eager_result, compiled_result), "Eager and Compiled results differ"
print("Test passed.")