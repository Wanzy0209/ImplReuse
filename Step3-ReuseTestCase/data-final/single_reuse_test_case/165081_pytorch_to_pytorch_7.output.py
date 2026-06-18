import torch

# Reproduce the environment configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch.manual_seed(52676)

# Ensure CUDA is available as the bug report specifies device=cuda
if not torch.cuda.is_available():
    print("CUDA is not available. This test requires CUDA to run as per the bug report context.")
    exit()

def test_clamp_min_dynamo():
    # Recreating the tensor setup from the fuzzer output
    # arg_0: size=(9, 9, 9), dtype=float64, device=cuda
    arg_0 = torch.randn(9, 9, 9, dtype=torch.float64, device='cuda')
    # arg_1: size=(9, 9, 11), dtype=float64, device=cuda
    arg_1 = torch.randn(9, 9, 11, dtype=torch.float64, device='cuda')

    # Operations from the fuzzer program
    var_node_5 = torch.matmul(arg_0, arg_1)

    # Adaptation: Replace the original torch.nonzero call with torch.clamp_min
    # We clamp the values to a minimum of 0.0
    result = torch.clamp_min(var_node_5, 0.0)

    return result

# Run in Eager mode
print("Running Eager mode...")
eager_output = test_clamp_min_dynamo()

# Run in Compiled mode (torch._dynamo)
print("Running Compiled mode...")
compiled_fn = torch.compile(test_clamp_min_dynamo)
compiled_output = compiled_fn()

# Verify results match
if torch.allclose(eager_output, compiled_output):
    print("SUCCESS: Eager and Compiled outputs match for torch.clamp_min.")
else:
    print("FAILURE: Eager and Compiled outputs differ.")
    print("Max diff:", torch.max(torch.abs(eager_output - compiled_output)))