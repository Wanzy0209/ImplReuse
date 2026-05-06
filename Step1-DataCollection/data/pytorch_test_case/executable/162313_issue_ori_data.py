import torch

flag = True


@torch.compile(backend="eager")
def fn(x):
    x = x + 1
    torch._dynamo.graph_break()
    x = x + 2
    if flag:
        with torch.no_grad():
            torch._dynamo.graph_break()
    else:
        with torch.no_grad():
            torch._dynamo.graph_break()
    return x + 4


fn(torch.ones(3))
flag = False
fn(torch.ones(3))