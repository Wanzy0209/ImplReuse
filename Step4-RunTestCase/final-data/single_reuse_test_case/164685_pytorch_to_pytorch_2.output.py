import torch

# Fix: Check if torch._dynamo exists before accessing its attributes
# to prevent AttributeError in environments where it is not available.
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(19989)

def fuzzed_program(arg_0, sentinel):
    var_node_2 = -6 # dtype=int64
    var_node_3 = arg_0 # dtype=int32
    
    # Adaptation: Replace direct multiplication with torch.prod
    # Original: var_node_1 = var_node_2 * var_node_3
    # We create a tensor containing the operands and compute the product.
    # This tests the behavior of torch.prod within the compiled graph.
    input_tensor = torch.tensor([var_node_2, var_node_3], dtype=torch.int32)
    var_node_1 = torch.prod(input_tensor)
    
    var_node_5 = torch.full((), 1, dtype=torch.int64) # size=(), stride=(), dtype=int64, device=cuda
    var_node_4 = var_node_5.item() # dtype=int64
    var_node_0 = var_node_1 / var_node_4 # dtype=int64
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.tensor(torch.randn(()), dtype=torch.int32).item()

args = (arg_0,) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')