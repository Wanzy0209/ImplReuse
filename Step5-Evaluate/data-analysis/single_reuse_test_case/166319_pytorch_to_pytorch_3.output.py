import torch
import torch.library

# Check for CUDA availability as the original issue was device-specific
if not torch.cuda.is_available():
    print("Skipping test: CUDA not available")
    exit(0)

# Fix: Check if _dynamo exists before accessing it to handle environment variations
if hasattr(torch, '_dynamo'):
    torch._dynamo.config.capture_scalar_outputs = True
else:
    print("Warning: torch._dynamo not found, skipping config.")

torch.manual_seed(1215252001)

# Define the custom operator name
qualname = "test_ns::fuzzed_op"

# Define the operator signature
def schema(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    pass

torch.library.define(qualname, schema)

# --- Abstract Implementation (Meta Kernel) ---
# This is the API under test: torch.library.impl_abstract
# It defines the behavior of the operator on FakeTensors (metadata only).
@torch.library.impl_abstract(qualname)
def fuzzed_op_meta(a, b, c, s):
    var_node_4 = a
    var_node_3 = torch.squeeze(var_node_4)
    var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0]
    var_node_1 = torch.squeeze(var_node_2)
    
    var_node_7 = b
    var_node_8 = c
    _input_size_var_node_6 = var_node_7.size(0)
    
    # In meta kernel, randint returns a FakeTensor with the correct shape
    _index_var_node_6 = torch.randint(0, _input_size_var_node_6, (18, 15), device=var_node_7.device)
    
    var_node_6 = torch.gather(var_node_7, 0, _index_var_node_6)
    var_node_5 = torch.chunk(var_node_6, 2, dim=0)[0]
    var_node_0 = torch.mul(var_node_1, var_node_5)
    
    result = var_node_0 * s
    if result.is_complex():
        result = result.real
    return result

# --- Concrete Implementation (Eager Kernel) ---
# This defines the actual runtime behavior.
@torch.library.impl(qualname, "CompositeExplicitAutograd")
def fuzzed_op_impl(a, b, c, s):
    var_node_4 = a
    var_node_3 = torch.squeeze(var_node_4)
    var_node_2 = torch.chunk(var_node_3, 4, dim=2)[0]
    var_node_1 = torch.squeeze(var_node_2)
    
    var_node_7 = b
    var_node_8 = c
    _input_size_var_node_6 = var_node_7.size(0)
    _index_var_node_6 = torch.randint(0, _input_size_var_node_6, (18, 15), device=var_node_7.device)
    
    var_node_6 = torch.gather(var_node_7, 0, _index_var_node_6)
    var_node_5 = torch.chunk(var_node_6, 2, dim=0)[0]
    var_node_0 = torch.mul(var_node_1, var_node_5)
    
    result = var_node_0 * s
    if result.is_complex():
        result = result.real
    return result

# --- Test Setup ---

# Prepare inputs matching the bug report's specific strides and dtypes
sentinel = torch.tensor(1.0, requires_grad=True, device="cuda")
arg_0 = torch.as_strided(torch.randint(5, 30, (484,), device="cuda").to(torch.int32), (9, 1, 15, 4), (60, 60, 0, 1))
arg_1 = torch.as_strided(torch.randint(5, 30, (300,), device="cuda").to(torch.int64), (20, 15), (15, 1))
arg_2 = torch.as_strided(torch.randint(5, 30, (270,), device="cuda").to(torch.int64), (18, 15), (15, 1))

def run_test():
    # 1. Test Eager Execution
    print("Testing eager execution with custom op...")
    try:
        result_eager = torch.ops.test_ns.fuzzed_op(arg_0, arg_1, arg_2, sentinel)
        print(" Eager success")
    except Exception as e:
        print(f" Eager failed: {e}")
        return

    # 2. Test Compiled Execution
    # This exercises the torch.library.impl_abstract (meta kernel) during tracing
    print("Testing compiled execution with custom op...")
    def compiled_func(a, b, c, s):
        return torch.ops.test_ns.fuzzed_op(a, b, c, s)

    try:
        compiled_program = torch.compile(compiled_func, fullgraph=True, dynamic=True)
        result_compiled = compiled_program(arg_0, arg_1, arg_2, sentinel)
        print(" Compile success")
    except Exception as e:
        print(f" Compile failed: {e}")
        raise

if __name__ == "__main__":
    run_test()