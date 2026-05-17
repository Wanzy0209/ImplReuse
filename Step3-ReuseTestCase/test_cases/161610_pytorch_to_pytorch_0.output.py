import torch
from typing import NamedTuple

# Define the NamedTuple structure
class MyNamedTuple(NamedTuple):
    first: torch.Tensor
    second: torch.Tensor

# Define a subclass to test dynamic attribute persistence
class MyNamedTupleSubclass(MyNamedTuple):
    pass

# The function to be tested, which adds a dynamic attribute
def fn(tup: MyNamedTuple) -> MyNamedTuple:
    extra_info = torch.tensor(4.0)
    tup.extra_info = extra_info  # Add dynamic attribute
    return tup

def test_namedtuple_dynamic_attribute_persistence():
    """
    Test that dynamic attributes added to NamedTuple subclasses 
    are preserved when using torch.compile with the eager backend.
    """
    
    # 1. Test Eager Execution (Baseline)
    print("Testing Eager Execution:")
    input_tup = MyNamedTupleSubclass(first=torch.tensor([2.0]), second=torch.tensor(1.0))
    eager_result = fn(input_tup)
    
    # Verify eager result has the attribute
    assert hasattr(eager_result, 'extra_info'), "Eager execution failed to set attribute"
    assert torch.equal(eager_result.extra_info, torch.tensor(4.0)), "Eager execution attribute value mismatch"
    print(f"Eager Result: {eager_result.extra_info}")

    # 2. Test Compiled Execution (backend="eager")
    print("\nTesting Compiled Execution (backend='eager'):")
    compiled_fn = torch.compile(fn, backend="eager")
    
    # Reset input
    input_tup_compiled = MyNamedTupleSubclass(first=torch.tensor([2.0]), second=torch.tensor(1.0))
    compiled_result = compiled_fn(input_tup_compiled)

    # Verify compiled result preserves the attribute
    # This assertion will fail if the bug is present
    try:
        assert hasattr(compiled_result, 'extra_info'), "Compiled execution failed to preserve attribute"
        assert torch.equal(compiled_result.extra_info, torch.tensor(4.0)), "Compiled execution attribute value mismatch"
        print(f"Compiled Result: {compiled_result.extra_info}")
    except AttributeError as e:
        print(f"Bug reproduced: {e}")
        raise

if __name__ == "__main__":
    test_namedtuple_dynamic_attribute_persistence()
    print("\nTest passed successfully!")