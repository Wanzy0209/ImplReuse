import torch

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