import torch
from typing import NamedTuple

class MyNamedTuple(NamedTuple):
    first: torch.Tensor
    second: torch.Tensor

class MyNamedTupleSubclass(MyNamedTuple):
    pass

# Setup from the bug report: Create a NamedTuple subclass instance with a dynamic attribute
extended_tup = MyNamedTupleSubclass(first=torch.tensor([2.0]), second=torch.tensor(1.0))
extended_tup.extra_info = torch.tensor(4.0)  # Add dynamic attribute

# Test torch.all with the NamedTuple subclass
# torch.all expects a Tensor input, not a tuple. We must convert the tuple of tensors to a single tensor.
# We flatten the tensors to ensure shape compatibility (e.g., [2.0] vs 1.0) before stacking.
print("Testing torch.all with NamedTuple subclass:")
stacked_tensor = torch.stack([t.flatten() for t in extended_tup])
result = torch.all(stacked_tensor)
print(f"torch.all result: {result}")

# Expected behavior: Both tensors are non-zero, so the result should be True
assert result == True, "torch.all should return True for non-zero tensors in the tuple"