import torch

# Reproduce the configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

def test_nonzero_behavior(input_tensor):
    """
    Test function that calls torch.nonzero.
    The bug report identified a divergence in eager vs compiled mode
    specifically regarding stride hints and shape handling.
    """
    # The call site for the API under test
    result = torch.nonzero(input_tensor)
    
    # Perform a subsequent operation to ensure the tensor is used,
    # similar to the original bug report where nonzero output was added to another tensor.
    # This helps expose stride/shape mismatches.
    if result.numel() > 0:
        return result + 1
    return result

# Setup input: A boolean tensor with all False values.
# This results in an empty tensor from nonzero, which is the edge case triggering the divergence.
input_tensor = torch.full((3, 4), False, dtype=torch.bool)

# 1. Run in Eager mode
try:
    eager_result = test_nonzero_behavior(input_tensor)
    print(f" Eager execution success. Shape: {eager_result.shape}, Stride: {eager_result.stride()}")
except Exception as e:
    print(f" Eager execution failed: {e}")
    eager_result = None

# 2. Run in Compiled mode
try:
    compiled_fn = torch.compile(test_nonzero_behavior, fullgraph=True, dynamic=True)
    compiled_result = compiled_fn(input_tensor)
    print(f" Compiled execution success. Shape: {compiled_result.shape}, Stride: {compiled_result.stride()}")
except Exception as e:
    print(f" Compiled execution failed: {e}")
    compiled_result = None

# 3. Verify Consistency
if eager_result is not None and compiled_result is not None:
    if torch.equal(eager_result, compiled_result):
        print(" Results match: Eager and Compiled outputs are identical.")
    else:
        print(f" Divergence detected: Eager and Compiled outputs differ.")
        print(f"   Eager: {eager_result}")
        print(f"   Compiled: {compiled_result}")