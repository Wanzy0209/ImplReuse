import torch
from torch.nn.attention.flex_attention import flex_attention

# Configuration from the bug report
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10):
    t0 = arg0 # size=(27, 26, 62, 122)
    t1 = arg1 # size=(27, 26, 124, 122)
    t2 = arg2 # size=(27, 26, 124, 122)
    t3 = flex_attention(t0, t1, t2)
    t4 = arg3 # size=(27, 26, 124, 122)
    t5 = arg4 # size=(27, 26, 248, 122)
    t6 = arg5 # size=(27, 26, 248, 122)
    t7 = flex_attention(t4, t5, t6)
    t8 = flex_attention(t3, t7, t7)
    t9 = arg6 # size=(27, 26, 31, 122)
    t10 = arg7 # size=(27, 26, 124, 122)
    t11 = flex_attention(t9, t7, t10)
    t12 = flex_attention(t11, t8, t3)
    t13 = arg8 # size=(27, 26, 31, 122)
    t14 = arg9 # size=(27, 26, 124, 122)
    t15 = flex_attention(t13, t2, t14)
    t16 = arg10 # size=(27, 26, 124, 122)
    t17 = t16.clone(); t17.zero_()
    t18 = flex_attention(t17, t8, t3)
    t19 = flex_attention(t15, t17, t18)
    t20 = flex_attention(t8, t12, t19)
    output = t20
    return output

# Setup arguments
# Replacing torch.rand with torch.fliplr(torch.ones(...)) as requested.
# torch.fliplr requires an input tensor, so we generate one with torch.ones.
# We maintain the specific shapes and device requirements from the original test case.
arg0 = torch.fliplr(torch.ones([27, 26, 62, 122], dtype=torch.float32, device='cuda', requires_grad=True))
arg1 = torch.fliplr(torch.ones([27, 26, 124, 122], dtype=torch.float32, device='cuda', requires_grad=True))
arg2 = torch.fliplr(torch.ones([27, 26, 124, 122], dtype=torch.float32, device='cuda', requires_grad=True))
arg3 = torch.fliplr(torch.ones([27, 26, 124, 122], dtype=torch.float32, device='cuda', requires_grad=True))
arg4 = torch.fliplr(torch.ones([27, 26, 248, 122], dtype=torch.float32, device='cuda', requires_grad=True))
arg5 = torch.fliplr(torch.ones([27, 26, 248, 122], dtype=torch.float32, device='cuda', requires_grad=True))
arg6 = torch.fliplr(torch.ones([27, 26, 31, 122], dtype=torch.float32, device='cuda', requires_grad=True))
arg7 = torch.fliplr(torch.ones([27, 26, 124, 122], dtype=torch.float32, device='cuda', requires_grad=True))
arg8 = torch.fliplr(torch.ones([27, 26, 31, 122], dtype=torch.float32, device='cuda', requires_grad=True))
arg9 = torch.fliplr(torch.ones([27, 26, 124, 122], dtype=torch.float32, device='cuda', requires_grad=True))
arg10 = torch.fliplr(torch.ones([27, 26, 124, 122], dtype=torch.float32, device='cuda', requires_grad=True))

if torch.cuda.is_available():
    # Verify the similar API: torch.compile
    # The bug report indicates an OOM occurs in compile mode.
    compiled_foo = torch.compile(foo)
    
    try:
        # Run the compiled function
        output = compiled_foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
        print("Test passed: Compiled execution succeeded without OOM.")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Bug reproduced: OOM occurred during compiled execution. Error: {e}")
        else:
            raise
else:
    print("CUDA is not available. Skipping test.")