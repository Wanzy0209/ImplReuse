import torch

# Fix for ImportError: impl_abstract is not available in older PyTorch versions (e.g., 2.0)
# It was introduced in 2.1, previously known as register_fake.
try:
    from torch.library import Library, impl_abstract
except ImportError:
    try:
        from torch.library import Library, register_fake
        impl_abstract = register_fake
    except ImportError:
        # If neither exists, define a dummy to allow import, though functionality may be limited
        from torch.library import Library
        def impl_abstract(name):
            def decorator(func):
                return func
            return decorator

# Configure Dynamo settings as per the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch.manual_seed(19989)

# Define a custom library and operator to encapsulate the logic from the bug report.
# This allows us to explicitly test the registration of the abstract implementation.
lib = Library("test_div_lib", "DEF")
lib.define("custom_div(Tensor x, int y) -> Tensor")

# Register the abstract implementation using the similar API: torch.library.impl_abstract
# This defines the behavior for FakeTensors (used during compilation/tracing).
@impl_abstract("test_div_lib::custom_div")
def custom_div_abstract(x, y):
    # The bug involves an int32 tensor divided by an int64 scalar, resulting in an int64 tensor.
    # We explicitly define the output shape and dtype here to ensure the compiler
    # has the correct metadata, addressing the potential divergence.
    return x.new_empty(x.shape, dtype=torch.int64)

# Register the concrete (eager) implementation
@lib.impl("custom_div", "CompositeExplicitAutograd")
def custom_div_impl(x, y):
    return x / y

def fuzzed_program(arg_0, sentinel):
    var_node_2 = -6 # dtype=int64
    var_node_3 = arg_0 # dtype=int32
    var_node_1 = var_node_2 * var_node_3 # dtype=int32
    var_node_5 = torch.full((), 1, dtype=torch.int64)
    var_node_4 = var_node_5.item() # dtype=int64
    
    # Replace the standard division operator with our custom operator
    # which relies on the torch.library.impl_abstract registration.
    var_node_0 = torch.ops.test_div_lib.custom_div(var_node_1, var_node_4)
    
    # Ensure gradient computation by multiplying with sentinel and taking real part
    result = var_node_0 * sentinel
    if result.is_complex():
        result = result.real
    return result

# Sentinel tensor to ensure gradient computation
sentinel = torch.tensor(1.0, requires_grad=True)

arg_0 = torch.tensor(torch.randn(()), dtype=torch.int32).item()

args = (arg_0,) + (sentinel,)

# Test Eager mode
result_original = fuzzed_program(*args)
print(' eager success')

# Test Compile mode
# This will utilize the abstract implementation registered via torch.library.impl_abstract
compiled_program = torch.compile(fuzzed_program, fullgraph=True, dynamic=True)
result_compiled = compiled_program(*args)
print(' compile success')

# Verify consistency
assert torch.equal(result_original, result_compiled), "Divergence between eager and compiled results"