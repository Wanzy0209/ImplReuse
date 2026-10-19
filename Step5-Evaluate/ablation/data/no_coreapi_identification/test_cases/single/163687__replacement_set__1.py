import torch
from torch.nn.attention.flex_attention import flex_attention

# Configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10):
    t0 = arg0
    t1 = arg1
    t2 = arg2
    t3 = flex_attention(t0, t1, t2)
    t4 = arg3
    t5 = arg4
    t6 = arg5
    t7 = flex_attention(t4, t5, t6)
    t8 = flex_attention(t3, t7, t7)
    t9 = arg6
    t10 = arg7
    t11 = flex_attention(t9, t7, t10)
    t12 = flex_attention(t11, t8, t3)
    t13 = arg8
    t14 = arg9
    t15 = flex_attention(t13, t2, t14)
    t16 = arg10
    t17 = t16.clone(); t17.zero_()
    t18 = flex_attention(t17, t8, t3)
    t19 = flex_attention(t15, t17, t18)
    t20 = flex_attention(t8, t12, t19)
    output = t20
    return output

# Helper to generate inputs using torch.flipud instead of torch.rand
def make_input(shape):
    # Create a base tensor and apply torch.flipud as requested
    return torch.flipud(torch.randn(shape, dtype=torch.float32, device='cuda', requires_grad=True))

# Create arguments
arg0 = make_input([27, 26, 62, 122])
arg1 = make_input([27, 26, 124, 122])
arg2 = make_input([27, 26, 124, 122])
arg3 = make_input([27, 26, 124, 122])
arg4 = make_input([27, 26, 248, 122])
arg5 = make_input([27, 26, 248, 122])
arg6 = make_input([27, 26, 31, 122])
arg7 = make_input([27, 26, 124, 122])
arg8 = make_input([27, 26, 31, 122])
arg9 = make_input([27, 26, 124, 122])
arg10 = make_input([27, 26, 124, 122])

# Run eager
out_eager = foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)

# Run compiled (Testing torch.compile)
compiled_foo = torch.compile(foo)
out_compiled = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)

# Check consistency
assert torch.allclose(out_eager, out_compiled, atol=1e-3)