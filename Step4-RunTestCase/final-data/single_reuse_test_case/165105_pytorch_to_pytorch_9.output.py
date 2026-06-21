import torch

# Safe import and configuration for torch._dynamo
# If the module is not found (e.g., in older PyTorch versions), we skip the specific config.
try:
    import torch._dynamo
    # Configuration from the original bug report
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
except ImportError:
    print("Warning: torch._dynamo module not found. Proceeding with default configuration.")

torch.manual_seed(70609)

# Determine device (original bug report used cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_narrow_api():
    # Setup tensors similar to the original context (var_node_9)
    # Using float16 and specific shapes found in the bug report
    tensor_a = torch.full((6, 13), 1.3154296875, dtype=torch.float16, device=device)

    # Original call site: torch.matmul(var_node_9.to(torch.float16), var_node_10.to(torch.float16))
    # Adapted call site: torch.narrow(...)
    # We test torch.narrow on the tensor to verify behavior in this context.
    # Narrowing along dimension 0, starting at 1, length 4.
    result = torch.narrow(tensor_a, 0, 1, 4)

    return result

# Run in eager mode
eager_output = test_narrow_api()

# Check if torch.compile is available (it requires torch._dynamo internally)
if hasattr(torch, 'compile'):
    # Run in compiled mode (checking for divergence as per the bug title)
    compiled_fn = torch.compile(test_narrow_api)
    compiled_output = compiled_fn()

    # Verify results match
    assert torch.equal(eager_output, compiled_output), "Eager and compiled outputs diverged for torch.narrow"
    print("Test passed: torch.narrow behaves consistently in eager and compiled modes.")
else:
    print("torch.compile is not available in this environment. Skipping compiled mode comparison.")