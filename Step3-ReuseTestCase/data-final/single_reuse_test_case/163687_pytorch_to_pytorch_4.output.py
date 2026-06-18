import torch
import torch._dynamo
import torch._inductor

# Replicate the configuration from the bug report to ensure the same environment
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.capture_dynamic_output_shape_ops = True
torch._inductor.config.emulate_precision_casts = True

# Define a function using the similar API (torch.all)
# Adapted from the original 'foo' which used flex_attention.
# Since torch.all is a reduction operation, we adapt the logic to test its behavior
# within the compile context, rather than trying to fit it into the complex
# flex_attention graph structure which expects specific tensor shapes.
def foo(t0):
    # Original call site: t3 = flex_attention(t0, t1, t2)
    # Adapted call site: Using torch.all to verify eager/compile consistency
    return torch.all(t0 > 0)

# Setup inputs mimicking the original test case's tensor properties
# size=(27, 26, 62, 122), dtype=float32, device=cuda
arg0 = torch.rand([27, 26, 62, 122], dtype=torch.float32, device='cuda')

# Run in eager mode
eager_result = foo(arg0)

# Run in compiled mode
compiled_foo = torch.compile(foo)
compiled_result = compiled_foo(arg0)

# Verify that the results match to check for eager/compile divergence
assert eager_result == compiled_result, f"Eager and Compile results diverge: {eager_result} vs {compiled_result}"

print("Test passed: torch.all behaves consistently in eager and compile modes.")