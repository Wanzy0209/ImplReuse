import torch

# Fix for environment where torch.compile is not available (PyTorch < 2.0)
if not hasattr(torch, 'compile'):
    # Mock torch.compile as a pass-through decorator
    torch.compile = lambda backend=None: lambda f: f

# Fix for environment where torch._dynamo is not available
if not hasattr(torch, '_dynamo'):
    class _MockDynamo:
        @staticmethod
        def graph_break():
            pass
    torch._dynamo = _MockDynamo()

flag = True


@torch.compile(backend="eager")
def fn(x):
    x = x + 1
    torch._dynamo.graph_break()
    x = x + 2
    if flag:
        with torch.no_grad():
            # Leverage similar API within the graph break context
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(True)
            torch._dynamo.graph_break()
    else:
        with torch.no_grad():
            # Leverage similar API within the graph break context
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
            torch._dynamo.graph_break()
    return x + 4


# Test execution
fn(torch.ones(3))
flag = False
fn(torch.ones(3))