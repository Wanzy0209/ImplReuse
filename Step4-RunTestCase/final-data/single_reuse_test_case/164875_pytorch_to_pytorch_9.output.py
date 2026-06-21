import torch
import torch.testing
import sys

# Fix: Check if torch._dynamo is available before accessing it
if not hasattr(torch, '_dynamo'):
    print("Skipping test: torch._dynamo is not available in this PyTorch version.")
    sys.exit(0)

# Configure Dynamo settings as per the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

torch.manual_seed(1014698)

def fuzzed_program(arg_0, sentinel):
    # arg_0: size=(20, 0), stride=(1, 20), dtype=int64, device=cuda
    # Adaptation: Replace torch.add with torch.sigmoid
    # The original bug involved shape (20, 0). We test if sigmoid handles this shape correctly in eager vs compile.
    var_node_0 = torch.sigmoid(arg_0)
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

# Recreate the specific input shape (20, 0) that triggered the divergence
arg_0 = torch.as_strided(torch.randint(5, 30, (20,)).to(torch.int64), (20, 0), (1, 20))

args = (arg_0,) + (sentinel,)

# Run in eager mode
result_original = fuzzed_program(*args)
print(' eager success')

# Run in compiled mode
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')

# Assert that the results match to catch any divergence
torch.testing.assert_close(result_original, result_compiled)