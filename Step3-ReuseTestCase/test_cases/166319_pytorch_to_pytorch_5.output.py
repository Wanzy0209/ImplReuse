import torch

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch.manual_seed(1215252001)

# Setup tensors with specific strides and dtypes from the bug report
arg_0 = torch.as_strided(torch.randint(5, 30, (484,)).to(torch.int32), (9, 1, 15, 4), (60, 60, 0, 1))
arg_1 = torch.as_strided(torch.randint(5, 30, (300,)).to(torch.int64), (20, 15), (15, 1))
arg_2 = torch.as_strided(torch.randint(5, 30, (270,)).to(torch.int64), (18, 15), (15, 1))

def test_func(arg_0, arg_1, arg_2):
    # Reproduce the tensor manipulation logic from the original bug report
    var_node_4 = arg_0
    var_node_3 = torch.squeeze(var_node_4)
    var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0]
    var_node_1 = torch.squeeze(var_node_2)
    
    var_node_7 = arg_1
    var_node_8 = arg_2
    _input_size_var_node_6 = var_node_7.size(0)
    _index_var_node_6 = torch.randint(0, _input_size_var_node_6, (18, 15), device=var_node_7.device)
    var_node_6 = torch.gather(var_node_7, 0, _index_var_node_6)
    var_node_5 = torch.chunk(var_node_6, 2, dim=0)[0]
    
    var_node_0 = torch.mul(var_node_1, var_node_5)
    
    # Adaptation: Replace the original logic with torch.any
    # The original bug involved scalar outputs and lowering exceptions.
    # torch.any produces a scalar boolean, which is relevant to capture_scalar_outputs.
    return torch.any(var_node_0)

# Run eager mode
try:
    result_eager = test_func(arg_0, arg_1, arg_2)
    print(f' eager success: {result_eager}')
except Exception as e:
    print(f' eager failed: {e}')

# Run compiled mode
try:
    compiled_program = torch.compile(test_func, fullgraph=True, dynamic=True)
    result_compiled = compiled_program(arg_0, arg_1, arg_2)
    print(f' compile success: {result_compiled}')
    
    # Verify divergence
    assert result_eager == result_compiled, "Divergence detected between eager and compiled outputs"
except Exception as e:
    print(f' compile failed: {e}')