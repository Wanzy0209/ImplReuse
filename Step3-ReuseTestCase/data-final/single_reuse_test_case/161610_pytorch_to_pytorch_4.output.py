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
# torch.all should handle the iterable (tuple) correctly, checking the truthiness of its elements
print("Testing torch.all with NamedTuple subclass:")
result = torch.all(extended_tup)
print(f"torch.all result: {result}")

# Expected behavior: Both tensors are non-zero, so the result should be True
assert result == True, "torch.all should return True for non-zero tensors in the tuple"