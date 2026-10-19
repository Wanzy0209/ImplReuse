# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch

@torch.compile(backend="eager", fullgraph=True)
def fn(x):
    y = x + 10

    with torch._dynamo.set_fullgraph(False):
        class Foo:
            def __init__(self, x):
                self.x = x

    f = Foo(x)
    return f.x - y

x = torch.tensor([1.0])
y = fn(x)
print(y)