# SINGLE_LIBRARY_BASELINE for pytorch
# original source preserved below
import torch
from typing import NamedTuple

class MyNamedTuple(NamedTuple):
    first: torch.Tensor
    second: torch.Tensor

class MyNamedTupleSubclass(MyNamedTuple):
    pass

def fn(tup: MyNamedTuple) -> torch.Tensor:
    extra_info = torch.tensor(4.0)
    tup.extra_info = extra_info  # Add dynamic attribute
    return tup

print("\nTesting NamedTuple with __setattr__:")
extended_tup = MyNamedTupleSubclass(first=torch.tensor([2.0]), second=torch.tensor(1.0))
setattr_result = fn(extended_tup)
print(f"NamedTuple __setattr__ result: {setattr_result.extra_info}")


print("\nTesting NamedTuple with __setattr__:")
extended_tup = MyNamedTupleSubclass(first=torch.tensor([2.0]), second=torch.tensor(1.0))
setattr_result = torch.compile(fn, backend="eager")(extended_tup)
print(f"NamedTuple __setattr__ result: {setattr_result.extra_info}")