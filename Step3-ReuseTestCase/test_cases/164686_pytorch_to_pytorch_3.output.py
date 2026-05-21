import torch
import torch.library

# Setup from the original bug report to ensure similar environment
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True

# Define a custom operator that encapsulates the logic from the bug report
torch.library.define("test_lib::fuzzed_op", "(Tensor, Tensor, Tensor) -> Tensor")

# Register the abstract implementation using torch.library.impl_abstract
# This defines the behavior for FakeTensors (metadata inference) which is crucial for torch.compile
@torch.library.impl_abstract("test_lib::fuzzed_op")
def fuzzed_op_abstract(arg_0, arg_1, sentinel):
    # Based on the logic in the bug report, the output is a float32 scalar
    return torch.empty((), dtype=torch.float32, device=arg_0.device)

# Register the concrete implementation for CPU and CUDA
@torch.library.impl("test_lib::fuzzed_op", "CPU")
@torch.library.impl("test_lib::fuzzed_op", "CUDA")
def fuzzed_op_impl(arg_0, arg_1, sentinel):
    # Extract scalar values from inputs
    val_0 = arg_0.item()
    val_1 = arg_1.item()
    sentinel_val = sentinel.item()

    # Replicate the exact logic from the original fuzzed_program
    var_node_3 = torch.full((), 1.0, dtype=torch.float32)
    var_node_2 = var_node_3.item()
    var_node_5 = -3
    var_node_6 = val_0
    var_node_4 = var_node_5 + var_node_6
    var_node_1 = var_node_2 + var_node_4
    var_node_9 = 1
    var_node_10 = -10
    var_node_8 = var_node_9 / var_node_10
    var_node_12 = val_1
    var_node_13 = -5
    var_node_11 = var_node_12 / var_node_13
    var_node_7 = var_node_8 + var_node_11
    var_node_0 = var_node_1 * var_node_7
    result = var_node_0 * sentinel_val

    return torch.tensor(result, dtype=torch.float32)

# Test the implementation
def run_test():
    # Create inputs (using fixed values for reproducibility)
    arg_0 = torch.tensor(5, dtype=torch.int64)
    arg_1 = torch.tensor(2, dtype=torch.int64)
    sentinel = torch.tensor(1.0, requires_grad=True)

    # 1. Test Eager execution
    print("Testing eager execution...")
    res_eager = torch.ops.test_lib.fuzzed_op(arg_0, arg_1, sentinel)
    print(f"Eager Result: {res_eager}")

    # 2. Test Compiled execution
    # This verifies that the abstract implementation registered via torch.library.impl_abstract
    # is correct and allows torch.compile to succeed without divergence.
    print("Testing compiled execution...")
    compiled_op = torch.compile(torch.ops.test_lib.fuzzed_op, fullgraph=True, dynamic=True)
    res_compiled = compiled_op(arg_0, arg_1, sentinel)
    print(f"Compiled Result: {res_compiled}")

    # Assert results match
    assert torch.allclose(res_eager, res_compiled), "Eager and compiled results diverge!"
    print(" Test Passed: torch.library.impl_abstract works correctly with torch.compile")

if __name__ == "__main__":
    run_test()