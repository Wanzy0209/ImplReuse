import torch
from typing import NamedTuple

# Replacement: torch.Tensor -> torch.backends.cuda.cufft_plan_cache
# Note: torch.backends.cuda.cufft_plan_cache is a function, but used here as a type annotation
# and value source as per the replacement instruction.
class MyNamedTuple(NamedTuple):
    first: torch.backends.cuda.cufft_plan_cache
    second: torch.backends.cuda.cufft_plan_cache

class MyNamedTupleSubclass(MyNamedTuple):
    pass

def fn(tup: MyNamedTuple) -> torch.backends.cuda.cufft_plan_cache:
    # Replacement: torch.tensor -> torch.cummin
    # torch.cummin returns a named tuple (values, indices)
    input_tensor = torch.tensor([4.0, 3.0, 2.0])
    extra_info = torch.cummin(input_tensor, 0)
    tup.extra_info = extra_info  # Add dynamic attribute
    return tup

# Check for CUDA availability because the replacement API is CUDA-specific
if not torch.cuda.is_available():
    print("Skipping test: CUDA is not available for torch.backends.cuda.cufft_plan_cache")
else:
    print("\nTesting NamedTuple with __setattr__:")
    # Instantiate with the replacement API object
    cache_obj = torch.backends.cuda.cufft_plan_cache(0)
    extended_tup = MyNamedTupleSubclass(first=cache_obj, second=cache_obj)
    setattr_result = fn(extended_tup)
    assert hasattr(setattr_result, 'extra_info'), "Eager mode failed to persist attribute"
    print(f"NamedTuple __setattr__ result: {setattr_result.extra_info}")

    print("\nTesting NamedTuple with __setattr__ (compiled):")
    extended_tup = MyNamedTupleSubclass(first=cache_obj, second=cache_obj)
    # The bug is that torch.compile drops the attribute
    try:
        setattr_result = torch.compile(fn, backend="eager")(extended_tup)
        assert hasattr(setattr_result, 'extra_info'), "Compiled mode failed to persist attribute"
        print(f"NamedTuple __setattr__ result: {setattr_result.extra_info}")
    except AttributeError as e:
        print(f"Bug reproduced: {e}")