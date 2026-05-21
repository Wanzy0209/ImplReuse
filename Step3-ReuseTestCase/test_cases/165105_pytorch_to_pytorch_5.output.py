import torch
import torch.nn.functional as F

torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(70609)

def fuzzed_program(arg_0, arg_1, arg_2, sentinel):
    # Adaptation: Create a 4D tensor suitable for pixel_unshuffle
    # Original var_node_9 was (6, 13). We adapt it to (1, 6, 26, 26) to support downscale_factor=2.
    var_node_9 = torch.full((1, 6, 26, 26), 1.3154296875, dtype=torch.float16, device="cuda")
    
    # Original call site: torch.matmul(var_node_9.to(torch.float16), var_node_10.to(torch.float16))
    # Adapted call site: torch.nn.functional.pixel_unshuffle(var_node_9.to(torch.float16), 2)
    var_node_8 = F.pixel_unshuffle(var_node_9.to(torch.float16), 2)
    
    return var_node_8

# Setup dummy arguments to match the function signature
arg_0 = torch.randn(14, 6, dtype=torch.float16, device="cuda")
arg_1 = torch.randn(13, 1, dtype=torch.float16, device="cuda")
arg_2 = torch.randn(1, 10, dtype=torch.float16, device="cuda")

# Run Eager
eager_result = fuzzed_program(arg_0, arg_1, arg_2, None)

# Run Compiled
compiled_fn = torch.compile(fuzzed_program)
compiled_result = compiled_fn(arg_0, arg_1, arg_2, None)

# Verify equality to check for Divergence (DDE)
assert torch.allclose(eager_result, compiled_result)