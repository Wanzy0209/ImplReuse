import torch
import sys

# Check for CUDA availability as the original bug is specific to CUDA
if not torch.cuda.is_available():
    print("Test skipped: CUDA not available")
    sys.exit(0)

# Configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, sentinel):
    t0 = arg0
    t1 = arg1
    t2 = arg2
    t3 = arg3
    t4 = arg4
    t5 = torch.cat([t0, t1, t2, t3, t4], dim=2)
    t6 = t5.contiguous()
    t7 = arg5
    t8 = torch.exp(t7)
    
    # Using the similar API: torch.nn.functional.sigmoid
    # Replacing the original torch.rms_norm(t8, (62, 8)) call
    t9 = torch.nn.functional.sigmoid(t8)
    
    t10 = arg6
    t11 = torch.exp(t10)
    t12 = arg7
    t13 = torch.nn.functional.interpolate(t12, size=(127,), mode='nearest')
    t14 = torch.cat([t11, t13], dim=0)
    t15 = torch.baddbmm(t6, t9, t14)
    output = t15 + sentinel
    return output

# Input generation based on the provided test case
# Using bfloat16 and cuda as per the bug report
arg0 = torch.rand([93, 62, 23], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg1 = torch.rand([93, 62, 11], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg2 = torch.rand([93, 62, 10], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg3 = torch.rand([93, 62, 81], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg4 = torch.rand([93, 62, 2], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg5 = torch.rand([93, 62, 8], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg6 = torch.rand([77, 8, 127], dtype=torch.bfloat16, device='cuda', requires_grad=True)
arg7 = torch.rand([16, 8, 15], dtype=torch.bfloat16, device='cuda', requires_grad=True)
sentinel = torch.rand([93, 62, 127], dtype=torch.bfloat16, device='cuda', requires_grad=True)

# Run Eager
expected = foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, sentinel)

# Run Compiled
compiled_foo = torch.compile(foo)
actual = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, sentinel)

# Assertion to check for divergence
# Using a tolerance suitable for bfloat16 operations
assert torch.allclose(expected, actual, atol=1e-2, rtol=1e-2), "Divergence detected between eager and compiled modes"