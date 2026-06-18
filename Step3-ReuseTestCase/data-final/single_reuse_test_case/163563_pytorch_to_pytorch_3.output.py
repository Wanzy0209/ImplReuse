import torch
from torch.library import Library, impl_abstract

# Define a custom library and operator to test torch.library.impl_abstract
# We wrap the operation that was part of the original bug report (baddbmm)
my_lib = Library("my_custom_lib", "DEF")
my_lib.define("my_baddbmm(Tensor input, Tensor batch1, Tensor batch2) -> Tensor")

# Register the eager implementation for the custom operator
@my_lib.impl("my_baddbmm", "CUDA")
def my_baddbmm_impl(input, batch1, batch2):
    return torch.baddbmm(input, batch1, batch2)

# Register the abstract implementation using the target API: torch.library.impl_abstract
# This defines the behavior for FakeTensors during tracing/compilation.
@impl_abstract("my_custom_lib::my_baddbmm")
def my_baddbmm_abstract(input, batch1, batch2):
    # The abstract implementation must return a tensor with the correct shape and dtype
    # without performing the actual computation. We delegate to the standard op's meta function.
    return torch.baddbmm(input, batch1, batch2)

# Configuration from the original bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2):
    t0 = arg0
    t1 = torch.sigmoid(t0)
    t2 = arg1
    t3 = torch.sigmoid(t2)
    t4 = arg2
    t5 = torch.exp(t4)
    # Replace torch.baddbmm with the custom operator registered via impl_abstract
    t6 = torch.ops.my_custom_lib.my_baddbmm(t1, t3, t5)
    t7 = t6.reshape((193, 386, 459))
    return t7

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test.")
        exit(0)

    # Inputs from the original bug report
    arg0 = torch.rand([5699097, 6, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True)
    arg1 = torch.rand([5699097, 6, 256], dtype=torch.bfloat16, device='cuda', requires_grad=True)
    arg2 = torch.rand([5699097, 256, 1], dtype=torch.bfloat16, device='cuda', requires_grad=True)

    # Run eager mode
    out_eager = foo(arg0, arg1, arg2)
    out_eager.sum().backward()
    print('Eager Success! ')

    # Run compiled mode (torch.compile)
    # This tests if torch.compile correctly utilizes the abstract implementation
    # registered by torch.library.impl_abstract
    compiled_foo = torch.compile(foo, fullgraph=True, dynamic=True)
    out_compiled = compiled_foo(arg0, arg1, arg2)
    out_compiled.sum().backward()
    print('Compile Success! ')

    # Verify results match
    assert torch.allclose(out_eager, out_compiled, atol=1e-2, rtol=1e-2)
    print('Test Passed! ')