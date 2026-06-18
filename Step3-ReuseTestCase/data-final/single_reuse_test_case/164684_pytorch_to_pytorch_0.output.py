import torch

# Configure dynamo to capture scalar outputs and dynamic shapes
# This configuration is necessary to trigger the SymBool path during compilation
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

def test_mul_symbool_tensor(arg_0, sentinel):
    # Extract a scalar boolean value from the tensor
    # During tracing, this value becomes a SymBool
    scalar_bool = arg_0.squeeze().item()

    # Explicitly use torch.mul to test the API
    # Original code used the operator: result = var_node_0 * sentinel
    result = torch.mul(scalar_bool, sentinel)

    return result

# Setup inputs
# Create a boolean tensor
arg_0 = torch.randint(0, 2, (1,), dtype=torch.bool) > 0
# Create a float tensor with requires_grad=True
sentinel = torch.tensor(1.0, requires_grad=True)

# Run eager mode
print("Running eager mode...")
try:
    result_eager = test_mul_symbool_tensor(arg_0, sentinel)
    print(f"Eager result: {result_eager}")
except Exception as e:
    print(f"Eager failed: {e}")

# Run compiled mode
print("\nRunning compiled mode...")
try:
    compiled_fn = torch.compile(test_mul_symbool_tensor, fullgraph=True, dynamic=True)
    result_compiled = compiled_fn(arg_0, sentinel)
    print(f"Compiled result: {result_compiled}")
except Exception as e:
    print(f"Compiled failed: {e}")