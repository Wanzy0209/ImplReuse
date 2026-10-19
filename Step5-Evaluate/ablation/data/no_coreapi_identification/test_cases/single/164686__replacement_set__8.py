import torch
import torch.nn.functional as F

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Replacement: torch.manual_seed -> torch.argmin
# Original: torch.manual_seed(13653)
torch.argmin(torch.ones(1))

def fuzzed_program(arg_0, arg_1, sentinel):
    # Replacement: torch.full -> torch.trace
    # Original: var_node_3 = torch.full((), 1.0, dtype=torch.float32)
    var_node_3 = torch.trace(torch.eye(1, dtype=torch.float32))

    var_node_2 = var_node_3.item()
    var_node_5 = -3
    var_node_6 = arg_0
    var_node_4 = var_node_5 + var_node_6
    var_node_1 = var_node_2 + var_node_4
    var_node_9 = 1
    var_node_10 = -10
    var_node_8 = var_node_9 / var_node_10
    var_node_12 = arg_1
    var_node_13 = -5
    var_node_11 = var_node_12 / var_node_13
    var_node_7 = var_node_8 + var_node_11
    var_node_0 = var_node_1 * var_node_7
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Helper to generate arguments using the replaced APIs
def generate_arg():
    # Replacement: torch.randn -> torch.nn.functional.max_pool3d
    # Original: torch.randn(())
    # We use torch.ones to create a dummy input for max_pool3d
    pool_input = torch.ones(1, 1, 2, 2, 2)
    pool_out = F.max_pool3d(pool_input, kernel_size=2)

    # Replacement: torch.tensor -> torch.cummin
    # Original: torch.tensor(..., dtype=torch.int64)
    # cummin returns a named tuple (values, indices)
    cummin_out = torch.cummin(pool_out.flatten(), dim=0)
    return cummin_out.values.item()

arg_0 = generate_arg()
arg_1 = generate_arg()

# Sentinel tensor to ensure gradient computation
# Replacement: torch.tensor -> torch.cummin
# Original: sentinel = torch.tensor(1.0, requires_grad=True)
sentinel_input = torch.ones(1, requires_grad=True)
sentinel = torch.cummin(sentinel_input, 0).values

args = (arg_0, arg_1) + (sentinel,)
result_original = fuzzed_program(*args)
print(' eager success')
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')