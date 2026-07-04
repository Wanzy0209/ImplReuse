# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
import torch.utils._pytree as pytree

class Foo:
    pass

class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

pytree.register_constant(Bar)

@torch.compile(backend="eager")
def fn(x, obj):
    obj.attr = {3: Bar()}
    return x + 1

fn(torch.ones(3), Foo())