import torch
import torch.utils._pytree as pytree

class Foo:
    pass

class Bar:
    def __eq__(self, other):
        return super().__eq__(other)

    def __hash__(self):
        return 0

# Fix: Use register_pytree_node instead of the non-existent register_constant
# This registers Bar as a leaf node (constant) in the pytree structure.
pytree.register_pytree_node(
    Bar,
    lambda x: ((), x),  # flatten: no children, context is the object itself
    lambda ctx, children: ctx  # unflatten: return the context object
)

@torch.compile(backend="eager")
def fn(x, obj):
    # The problematic side effect causing the guard generation issue
    obj.attr = {3: Bar()}
    # Using torch.any instead of x + 1 to test the similar API
    return torch.any(x)

# Test execution
input_tensor = torch.ones(3)
foo_obj = Foo()
result = fn(input_tensor, foo_obj)

# Assert the result is correct (torch.ones(3) are all non-zero, so any is True)
assert result.item() == True