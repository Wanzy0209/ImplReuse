import torch
from typing import NamedTuple

class MyNamedTuple(NamedTuple):
    first: torch.Tensor
    second: torch.Tensor

class MyNamedTupleSubclass(MyNamedTuple):
    pass

# Setup from the bug report: Create the object and add a dynamic attribute
extended_tup = MyNamedTupleSubclass(first=torch.tensor([2.0]), second=torch.tensor([1.0]))
extended_tup.extra_info = torch.tensor(4.0)  # Add dynamic attribute

print("\nTesting torch.prod with NamedTuple containing dynamic attribute:")
# Adaptation: Replace torch.compile call with torch.prod
# We pass the tuple directly to verify if torch.prod handles the NamedTuple structure correctly
try:
    prod_result = torch.prod(extended_tup)
    print(f"torch.prod result: {prod_result}")
    
    # Verify the calculation: 2.0 * 1.0 = 2.0
    # Note: torch.prod converts the tuple to a tensor, effectively stacking the fields
    expected = torch.tensor(2.0)
    assert torch.allclose(prod_result, expected), f"Expected {expected}, but got {prod_result}"
    print("Assertion passed: torch.prod handled the NamedTuple correctly.")
except Exception as e:
    print(f"Error: {e}")
    raise