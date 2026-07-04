import torch
from torch.library import define, impl

# Define a custom operator to mimic flex_attention behavior
# This allows us to test the torch.library API
define("test_ns::custom_flex_attention", "(Tensor q, Tensor k, Tensor v) -> Tensor")

# Note: impl_abstract is removed as it is not available in the target PyTorch version.
# The concrete implementation below is sufficient for eager execution.

# Register a dummy concrete implementation for CPU
# This ensures the test runs on systems without CUDA
@impl("test_ns::custom_flex_attention", "CPU")
def custom_flex_attention_impl_cpu(q, k, v):
    # Return zeros to mimic execution without actual attention logic overhead
    return torch.zeros(q.shape[0], q.shape[1], q.shape[2], v.shape[3], 
                       dtype=q.dtype, device=q.device)

# Register a dummy concrete implementation for CUDA
@impl("test_ns::custom_flex_attention", "CUDA")
def custom_flex_attention_impl_cuda(q, k, v):
    # Return zeros to mimic execution without actual attention logic overhead
    return torch.zeros(q.shape[0], q.shape[1], q.shape[2], v.shape[3], 
                       dtype=q.dtype, device=q.device)

# Adapted function from the bug report
# Replaces flex_attention with torch.ops.test_ns.custom_flex_attention
def foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10):
    t0 = arg0
    t1 = arg1
    t2 = arg2
    t3 = torch.ops.test_ns.custom_flex_attention(t0, t1, t2)
    t4 = arg3
    t5 = arg4
    t6 = arg5
    t7 = torch.ops.test_ns.custom_flex_attention(t4, t5, t6)
    t8 = torch.ops.test_ns.custom_flex_attention(t3, t7, t7)
    t9 = arg6
    t10 = arg7
    t11 = torch.ops.test_ns.custom_flex_attention(t9, t7, t10)
    t12 = torch.ops.test_ns.custom_flex_attention(t11, t8, t3)
    t13 = arg8
    t14 = arg9
    t15 = torch.ops.test_ns.custom_flex_attention(t13, t2, t14)
    t16 = arg10
    t17 = t16.clone(); t17.zero_()
    t18 = torch.ops.test_ns.custom_flex_attention(t17, t8, t3)
    t19 = torch.ops.test_ns.custom_flex_attention(t15, t17, t18)
    t20 = torch.ops.test_ns.custom_flex_attention(t8, t12, t19)
    output = t20
    return output

# Setup inputs based on the bug report
device = 'cuda' if torch.cuda.is_available() else 'cpu'

arg0 = torch.rand([27, 26, 62, 122], dtype=torch.float32, device=device)
arg1 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device=device)
arg2 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device=device)
arg3 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device=device)
arg4 = torch.rand([27, 26, 248, 122], dtype=torch.float32, device=device)
arg5 = torch.rand([27, 26, 248, 122], dtype=torch.float32, device=device)
arg6 = torch.rand([27, 26, 31, 122], dtype=torch.float32, device=device)
arg7 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device=device)
arg8 = torch.rand([27, 26, 31, 122], dtype=torch.float32, device=device)
arg9 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device=device)
arg10 = torch.rand([27, 26, 124, 122], dtype=torch.float32, device=device)

# Run the test
print("Running test with torch.library.impl registered op...")
try:
    out = foo(arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10)
    print(f"Test passed. Output shape: {out.shape}")
    assert out.shape == (27, 26, 62, 122), f"Shape mismatch: expected (27, 26, 62, 122), got {out.shape}"
except Exception as e:
    print(f"Test failed with error: {e}")
    raise